import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence, pad_sequence, PackedSequence
from typing import List, Tuple
import random
from utils import *


def read_raw_dataset(path):
    cols = ['hash', 'trajectory_id', 'time_entry', 'time_exit',
            'x_entry', 'y_entry', 'x_exit', 'y_exit']
    df = pd.read_csv(path, usecols=cols)
    df[['x_entry', 'y_entry', 'x_exit', 'y_exit']] /= 1e6
    df['time_entry'] = pd.to_datetime(df['time_entry'], format="%H:%M:%S")
    df['time_exit'] = pd.to_datetime(df['time_exit'], format="%H:%M:%S")
    df['traj_id'] = df['trajectory_id'].str.split("_").str[-1].astype(float)
    return df


def train_eval_split(df, label, train_size, seed):
    if train_size <= 0 or train_size >= 1:
        raise ValueError(f"invalid train size {train_size}")
    device = df['hash'].unique()
    n = len(device)
    random.seed(seed)
    random.shuffle(device)
    train_device = device[:int(n*train_size)]
    eval_device = device[int(n*train_size):]
    df = df.set_index("hash")
    arc = pd.read_csv("arc_train.csv").set_index("hash")

    return (df.loc[train_device], df.loc[eval_device],
            label.loc[train_device], label.loc[eval_device],
            arc.loc[train_device], arc.loc[eval_device])


def load_train_eval_data(train_path, label_path, train_size,
                         x_center_bins, x_other_bins,
                         y_center_bins, y_other_bins,
                         t_near16_bins, t_other_bins, seed=10):
    x1 = 3.7509015068
    x2 = 3.7709015068
    y1 = -19.2689056133
    y2 = -19.2089056133

    df_train = read_raw_dataset(train_path)
    label = pd.read_csv(label_path, index_col="hash")
    df_train, df_eval, label_train, label_eval, arc_train, arc_eval = train_eval_split(df_train, label, train_size, seed)

    bins_x = get_bins(df_train[['x_entry', 'x_exit']].values.flatten(),
                      x_center_bins, x_other_bins, 3.7509015068, 3.7709015068)
    bins_y = get_bins(df_train[['y_entry', 'y_exit']].values.flatten(),
                      y_center_bins, y_other_bins, -19.2689056133, -19.2089056133)
    bins_t = get_time_bins(df_train[['time_entry', 'time_exit']].values.flatten(),
                           t_near16_bins, t_other_bins)

    df_train = merge_entry_exit(df_train)
    df_eval = merge_entry_exit(df_eval)

    bin_xyt_v2(df_train, bins_x, bins_y, bins_t)
    bin_xyt_v2(df_eval, bins_x, bins_y, bins_t)

    mu_x = df_train['x'].values.mean()
    std_x = df_train['x'].values.std()
    mu_y = df_train['y'].values.mean()
    std_y = df_train['y'].values.std()

    df_train['x'] = normalize(df_train['x'], mu_x, std_x)
    df_train['y'] = normalize(df_train['y'], mu_y, std_y)
    df_eval['x'] = normalize(df_eval['x'], mu_x, std_x)
    df_eval['y'] = normalize(df_eval['y'], mu_y, std_y)
    # df_test['x'] = normalize(df_test['x'], mu_x, std_x)
    # df_test['y'] = normalize(df_test['y'], mu_y, std_y)

    cross_product_transform(df_train, "x_bin", "y_bin",
                            len(bins_x) - 1, len(bins_y) - 1,
                            "xy_bin")
    cross_product_transform(df_train, "t_bin", "xy_bin",
                            len(bins_t) - 1, (len(bins_x) - 1)*(len(bins_y) - 1),
                            "xyt_bin")

    cross_product_transform(df_eval, "x_bin", "y_bin",
                            len(bins_x) - 1, len(bins_y) - 1,
                            "xy_bin")
    cross_product_transform(df_eval, "t_bin", "xy_bin",
                            len(bins_t) - 1, (len(bins_x) - 1)*(len(bins_y) - 1),
                            "xyt_bin")

    x1 = normalize(x1, mu_x, std_x)
    x2 = normalize(x2, mu_x, std_x)
    y1 = normalize(y1, mu_y, std_y)
    y2 = normalize(y2, mu_y, std_y)

    return (df_train, df_eval, label_train.sort_index(), label_eval.sort_index(),
            mu_x, mu_y, std_x, std_y,
            bins_x, bins_y, bins_t,
            x1, x2, y1, y2, arc_train.sort_index(), arc_eval.sort_index())


def load_test_data(path, mu_x, mu_y, std_x, std_y,
                   bins_x, bins_y, bins_t):

    df_test = read_raw_dataset(path)
    df_test = merge_entry_exit(df_test)
    bin_xyt_v2(df_test, bins_x, bins_y, bins_t)
    df_test['x'] = normalize(df_test['x'], mu_x, std_x)
    df_test['y'] = normalize(df_test['y'], mu_y, std_y)
    cross_product_transform(df_test, "x_bin", "y_bin",
                            len(bins_x) - 1, len(bins_y) - 1,
                            "xy_bin")
    cross_product_transform(df_test, "t_bin", "xy_bin",
                            len(bins_t) - 1, (len(bins_x) - 1)*(len(bins_y) - 1),
                            "xyt_bin")
    return df_test


class RNNDatasetMergeEntryExit(Dataset):
    def __init__(self, df, label, arc, dense_features, sparse_features):
        self.dense_features = dense_features
        self.sparse_features = sparse_features

        df = df.set_index("hash")
        if len(dense_features) > 0:
            self.dense = df[dense_features].groupby("hash").apply(lambda x: x.values).tolist()

        if len(sparse_features) > 0:
            self.sparse = df[sparse_features].groupby("hash").apply(lambda x: x.values).tolist()
        self.lengths = df.groupby("hash").apply(len).values[:, None]
        self.label = label['label'].values[:, None]
        self.arc = arc['lies_in_probability'].values[:, None]

    def info(self):
        return {'dense_features': self.dense_features,
                'sparse_features': self.sparse_features,
                'dense': len(self.dense) if len(self.dense_features) > 0 else "not used",
                'sparse': len(self.sparse) if len(self.sparse_features) > 0 else "not used",
                'arc': self.arc.shape,
                'label': self.label.shape}

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return (torch.from_numpy(self.dense[index]) if len(self.dense_features) > 0 else None,
                torch.from_numpy(self.sparse[index]) if len(self.sparse_features) > 0 else None,
                torch.from_numpy(self.lengths[index]),
                torch.from_numpy(self.arc[index]),
                torch.from_numpy(self.label[index]))


def collate_fn(batch: List[Tuple[Tensor]]):
    dense, sparse, length, arc, label = list(zip(*batch))
    dense = pad_sequence(dense, batch_first=True) if dense[0] is not None else None
    sparse = pad_sequence(sparse, batch_first=True) if sparse[0] is not None else None
    length = torch.tensor(length)
    arc = torch.stack(arc, dim=0)
    label = torch.stack(label, dim=0)
    return {"dense": dense, "sparse": sparse, "length": length, "label": label, "arc": arc}
