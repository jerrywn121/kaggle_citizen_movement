import torch
from trainer import Trainer
from config import config
from torch.utils.data import DataLoader
from data_loader import collate_fn, load_train_eval_data, RNNDatasetMergeEntryExit


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load data
    print(config.__dict__)
    print(f'reading and parsing data from {config.data_dir}')
    (df_train, df_eval, label_train, label_eval,
     mu_x, mu_y, std_x, std_y,
     bins_x, bins_y, bins_t,
     x1, x2, y1, y2, arc_train, arc_eval) = load_train_eval_data(f"{config.data_dir}/data_train.csv",
                                            f"{config.data_dir}/label.csv",
                                            config.train_size,
                                            config.x_center_bins, config.x_other_bins,
                                            config.y_center_bins, config.y_other_bins,
                                            config.t_near16_bins, config.t_other_bins,
                                            seed=10)

    # make dataset
    print('processing training set')
    dataset_train = RNNDatasetMergeEntryExit(df_train, label_train, arc_train,
                                             config.dense_features,
                                             config.sparse_features)
    print(dataset_train.info())

    print('processing eval set')
    dataset_eval = RNNDatasetMergeEntryExit(df_eval, label_eval, arc_eval,
                                            config.dense_features,
                                            config.sparse_features)
    print(dataset_eval.info())

    # make dataloader
    torch.manual_seed(0)
    print('loading train dataloader')
    dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size,
                                  shuffle=True, collate_fn=collate_fn, num_workers=config.n_cpu)
    print('loading eval dataloader')
    dataloader_eval = DataLoader(dataset_eval, batch_size=config.batch_size_test,
                                 shuffle=False, collate_fn=collate_fn, num_workers=config.n_cpu)

    # train
    trainer = Trainer(config)
    trainer.train(dataloader_train, dataloader_eval)
    print("======training done======\n")
