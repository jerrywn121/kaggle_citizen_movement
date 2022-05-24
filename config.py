import torch
from pathlib import Path


class Config:
    pass


config = Config()

# ----- trainer -----
config.n_cpu = 2
config.device = torch.device('cuda:0')  # torch.device("cpu")
config.batch_size_test = 512 * 2
config.batch_size = 512
config.use_multi_gpu_for_train = False
config.lr = 0.002
config.min_lr = 0.0001
config.weight_decay = 0
config.display_interval = 10000 // config.batch_size
config.num_epochs = 50
config.early_stopping = True
config.patience = 6
config.gradient_clipping = True
config.clipping_threshold = 1.

# ----- data -----
config.data_dir = Path('data')
config.chk_path = "checkpoint.chk"

config.dense_features = ['x', 'y']
config.sparse_features = ['x_bin', 'y_bin', 't_bin']

config.x_center_bins = 7
config.x_other_bins = 5
config.y_center_bins = 4
config.y_other_bins = 4
config.t_near16_bins = 10
config.t_other_bins = 7

num_x_bins = config.x_center_bins + 2 * config.x_other_bins
num_y_bins = config.y_center_bins + 2 * config.y_other_bins
num_t_bins = config.t_near16_bins + config.t_other_bins

# number of embeddings for sparse features
config.num_embs = {'x_bin': num_x_bins, 'y_bin': num_y_bins, 't_bin': num_t_bins,
                   'xy_bin': num_x_bins * num_y_bins,
                   'xyt_bin': num_x_bins * num_y_bins * num_t_bins}

config.train_size = 0.9
config.threshold = 0.5

# ----- model -----
config.num_layers = 4
config.hidden_size = [256] * config.num_layers

# all emb_dim should be the same (for convenience)
config.emb_dim = 20
