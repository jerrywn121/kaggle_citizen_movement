import pandas as pd
import numpy as np


def normalize(x, mu, std):
    return (x - mu) / std


def get_bins(data, num_center_bins, num_other_bins, low, high):
    mask_low = data < low
    mask_high = data > high
    bins_low = np.quantile(data[mask_low], q=np.linspace(0, 1, num_other_bins+1))
    bins_low[-1] = low
    bins_high = np.quantile(data[mask_high], q=np.linspace(0, 1, num_other_bins+1))
    bins_high[0] = high
    bins_center = np.quantile(data[(~mask_low) & (~mask_high)], q=np.linspace(0, 1, num_center_bins+1))
    return np.concatenate([bins_low, bins_center[1:-1], bins_high])


def get_time_bins(data, num_near16_bins, num_other_bins, threshold=pd.Timestamp('1900-01-01 14:50:00')):
    mask = data >= threshold
    bins_near16 = np.quantile(data[mask], q=np.linspace(0, 1, num_near16_bins+1))
    bins_other = np.quantile(data[~mask], q=np.linspace(0, 1, num_other_bins+1))
    bins_other[-1] = threshold
    return np.concatenate([bins_other, bins_near16[1:]])


def cut(col, bins):
    return pd.cut(col.clip(bins[0], bins[-1]), bins=bins, labels=False, include_lowest=True)


def bin_xyt_v2(df, bins_x, bins_y, bins_t):
    df['x_bin'] = cut(df['x'], bins_x)
    df['y_bin'] = cut(df['y'], bins_y)
    df['t_bin'] = cut(df['t'], bins_t)

    assert not np.isnan(df[['x_bin', 'y_bin', 't_bin']].values).any(), "nan value detected"


def bin_xyt(df, bins_x, bins_y, bins_t):
    df['x_entry_bin'] = cut(df['x_entry'], bins_x)
    df['x_exit_bin'] = cut(df['x_exit'], bins_x)
    df['y_entry_bin'] = cut(df['y_entry'], bins_y)
    df['y_exit_bin'] = cut(df['y_exit'], bins_y)
    df['t_entry_bin'] = cut(df['time_entry'], bins_t)
    df['t_exit_bin'] = cut(df['time_exit'], bins_t)

    assert not np.isnan(df[['x_entry_bin', 'x_exit_bin',
                            'y_entry_bin', 'y_exit_bin',
                            't_entry_bin',
                            't_exit_bin']].values).any(), "nan value detected"


def cross_product_transform(df, col1, col2, len_field1, len_field2, out_col):
    '''
    feature value should start with 0
    '''
    assert df[col1].max() < len_field1
    assert df[col2].max() < len_field2
    df[out_col] = df[col1] * len_field2 + df[col2]


def drop_last_row_in_group(df):
    lengths = df.groupby("hash").apply(len).cumsum()
    return df.drop(index=lengths.values - 1).reset_index(drop=True)


def merge_entry_exit(df):
    df = df.reset_index()
    exit = (df[['hash', 'trajectory_id', 'traj_id', 'x_exit', 'y_exit', 'time_exit']]
            .rename(columns={"x_exit": "x_entry",
                             "y_exit": "y_entry",
                             "time_exit": "time_entry"}))
    exit['traj_id'] += 0.5
    exit = pd.concat([df[['hash', 'trajectory_id', 'traj_id', 'x_entry', 'y_entry', 'time_entry']], exit], axis=0, ignore_index=True)
    exit = (exit.rename(columns={"x_entry": "x", "y_entry": "y", "time_entry": "t"})
                .sort_values(['hash', 'traj_id'])
                .reset_index(drop=True))
    return drop_last_row_in_group(exit)


def is_last_exit_in_center(df, x1, x2, y1, y2):
    last_exit = df.iloc[-1]
    return ((last_exit['x_exit'] > x1) &
            (last_exit['x_exit'] < x2) &
            (last_exit['y_exit'] > y1) &
            (last_exit['y_exit'] < y2))
