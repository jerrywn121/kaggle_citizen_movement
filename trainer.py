import torch
import torch.nn as nn
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import RNN, SimpleRNN
from torchmetrics import F1Score


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        torch.manual_seed(5)
        # self.model = RNN(config.hidden_size, config.num_layers,
        #                  config.dense_features, config.sparse_features,
        #                  config.num_embs, config.emb_dim)
        self.model = SimpleRNN(config.hidden_size[0], config.num_layers,
                               config.dense_features, config.sparse_features,
                               config.num_embs, config.emb_dim)
        if config.use_multi_gpu_for_train:
            print(f"will use {torch.cuda.device_count()} GPUs for training")
            self.model = nn.DataParallel(self.model).to(config.device)
        else:
            print(f"will use single GPU for training")
            self.model = self.model.to(config.device)
        self.opt = torch.optim.Adamax(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.lr_scheduler = ReduceLROnPlateau(self.opt, mode='max', factor=0.3, patience=2, verbose=True, min_lr=config.min_lr)
        self.f1_scorer = F1Score(num_classes=1, average='macro', threshold=config.threshold).to(self.device)
        self.bp_loss = nn.BCELoss()

    def score(self, y_pred, y_true, arc):
        with torch.no_grad():
            f1 = self.f1_scorer(y_pred, y_true)
        return f1

    def loss_func(self, y_pred, y_true):
        return self.bp_loss(y_pred, y_true.float())

    def retrieve(self, x):
        dense, sparse, length, label, arc = x['dense'], x['sparse'], x['length'], x['label'], x['arc']
        dense = dense.float().to(self.device) if dense is not None else None
        sparse = sparse.to(self.device) if sparse is not None else None
        # length = length
        arc = arc.float().to(self.device)
        label = x['label'][:, 0].to(self.device)
        return dense, sparse, length, arc, label

    def train_on_batch(self, x, return_score=False):
        self.model.train()
        dense, sparse, length, arc, label = self.retrieve(x)
        pred = self.model(dense, sparse, length, arc)
        self.opt.zero_grad()
        loss = self.loss_func(pred, label)
        loss.backward()
        if self.config.gradient_clipping:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clipping_threshold)
        self.opt.step()
        if return_score:
            return loss.item(), self.score(pred, label, arc).item()
        else:
            return loss.item()

    def evaluate(self, dataloader):
        self.model.eval()
        size = 0
        loss_sum = 0
        preds = []
        labels = []
        arcs = []
        with torch.no_grad():
            for x in dataloader:
                dense, sparse, length, arc, label = self.retrieve(x)
                pred = self.model(dense, sparse, length, arc)
                preds.append(pred)
                labels.append(label)
                arcs.append(arc)
                size += pred.size(0)
                loss_sum += (self.loss_func(pred, label).item() * pred.size(0))
            preds = torch.cat(preds, dim=0)
            arcs = torch.cat(arcs, dim=0)
            labels = torch.cat(labels, dim=0)
        return loss_sum / size, self.score(preds, labels, arcs).item()

    def train(self, dataloader_train, dataloader_eval):
        count = 0
        best = -1
        for i in range(self.config.num_epochs):
            print('\nepoch: {0}'.format(i + 1))
            # train
            # self.model.train()
            epoch_start_time = time.time()
            for j, x in enumerate(dataloader_train):
                if j % self.config.display_interval == 0:
                    loss_train, score_train = self.train_on_batch(x, return_score=True)
                    print('batch training loss: {:.5f}, score: {:.3f}'.format(loss_train, score_train))
                else:
                    _ = self.train_on_batch(x)

            # evaluation
            epoch_training_time = self.get_elapsed_time(epoch_start_time)
            eval_start_time = time.time()
            loss_eval, score_eval = self.evaluate(dataloader_eval)
            eval_time = self.get_elapsed_time(eval_start_time)
            print('epoch training time: {}\nepoch eval loss: {:.5f}, score: {:.3f}, eval time: {}'.format(epoch_training_time, loss_eval, score_eval, eval_time))
            self.lr_scheduler.step(score_eval)
            if score_eval <= best:
                count += 1
                print('eval score is not improved for {} epoch'.format(count))
            else:
                count = 0
                print('eval score is improved from {:.3f} to {:.3f}, saving model'.format(best, score_eval))
                self.save_model()
                best = score_eval

            if count == self.config.patience:
                print('early stopping reached, best score is {:3f}'.format(best))
                break

    def save_model(self):
        torch.save({'net': self.model.state_dict(),
                    'optimizer': self.opt.state_dict()}, self.config.chk_path)

    def get_elapsed_time(self, start_time):
        elapsed_time = time.time() - start_time
        return str(round(elapsed_time, 3)) + ' s' if elapsed_time < 60 else str(round(elapsed_time / 60, 3)) + "mins"
