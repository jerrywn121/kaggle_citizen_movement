import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from torch import Tensor
from typing import List, Dict, Optional


class RNN(nn.Module):
    def __init__(self, hidden_size: List[int], num_layers: int,
                 seq_dense_features: List[str], seq_sparse_features: List[str],
                 num_embs: Dict[str, int], emb_dim: int):
        super().__init__()
        assert len(hidden_size) == num_layers
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        input_size = len(seq_dense_features) + emb_dim * len(seq_sparse_features)
        if len(seq_sparse_features) > 0:
            self.sparse_emb = nn.ModuleList([nn.Embedding(num_embs[sp], emb_dim) for sp in seq_sparse_features])
        self.linear = nn.Linear(input_size, hidden_size[0])
        self.dropout = nn.Dropout(0.1)
        self.rnn_entry_exit = []
        self.rnn_exit_entry = []
        for i in range(num_layers):
            cur_input_size = hidden_size[0] if i == 0 else hidden_size[i-1]
            # input entry
            self.rnn_entry_exit.append(nn.RNNCell(cur_input_size, hidden_size[i]))
            # input exit
            self.rnn_exit_entry.append(nn.RNNCell(cur_input_size, hidden_size[i]))
        self.rnn_entry_exit = nn.ModuleList(self.rnn_entry_exit)
        self.rnn_exit_entry = nn.ModuleList(self.rnn_exit_entry)
        self.output = nn.Sequential(nn.Linear(hidden_size[-1], 1), nn.Sigmoid())

    def forward(self, dense: Optional[PackedSequence] = None, sparse: Optional[PackedSequence] = None,
                return_pred: bool = False) -> Tensor:
        if sparse is not None:
            batch_sizes = sparse.batch_sizes
            unsorted_indices = sparse.unsorted_indices
        else:
            batch_sizes = dense.batch_sizes
            unsorted_indices = dense.unsorted_indices
        t_max = len(batch_sizes)
        cur_sample_idx = 0
        h = [self._init_hidden_state(batch_sizes[0], self.hidden_size[i]) for i in range(self.num_layers)]

        output = []
        for t in range(t_max):
            for i in range(self.num_layers):
                if i == 0:
                    if sparse is not None:
                        batch_sparse = sparse.data[cur_sample_idx:cur_sample_idx + batch_sizes[t]]
                        batch_sparse = torch.cat([emb(batch_sparse[..., i]) for i, emb in enumerate(self.sparse_emb)], dim=-1)
                    if dense is not None:
                        batch_dense = dense.data[cur_sample_idx:cur_sample_idx + batch_sizes[t]]

                    if dense is None:
                        input_ = batch_sparse
                    elif sparse is None:
                        input_ = batch_dense
                    else:
                        input_ = torch.cat([batch_dense, batch_sparse], dim=-1)

                    input_ = F.relu(self.linear(input_))
                else:
                    input_ = h[i-1]

                if t % 2 == 0:
                    h[i] = self.rnn_entry_exit[i](input_, h[i])
                else:
                    h[i] = self.rnn_exit_entry[i](input_, h[i])

            if t < t_max - 1:
                if batch_sizes[t] > batch_sizes[t+1]:
                    for i in range(self.num_layers):
                        if i == self.num_layers - 1:
                            output.append(h[i][batch_sizes[t+1]:])
                        h[i] = h[i][:batch_sizes[t+1]]

            cur_sample_idx += batch_sizes[t]

        output.append(h[-1])
        output = torch.cat(output[::-1], dim=0)
        return self.output(output[unsorted_indices])[:, 0]

    def _init_hidden_state(self, batch_size: int, hidden_size: int) -> Tensor:
        return torch.zeros((batch_size, hidden_size)).to(torch.device("cuda:0"))


class SimpleRNN(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int,
                 seq_dense_features: List[str], seq_sparse_features: List[str],
                 num_embs: Dict[str, int], emb_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        input_size = len(seq_dense_features) + emb_dim * len(seq_sparse_features)
        if len(seq_sparse_features) > 0:
            self.sparse_emb = nn.ModuleList([nn.Embedding(num_embs[sp], emb_dim) for sp in seq_sparse_features])

        self.rnn = nn.RNN(input_size+1, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear_h = nn.Linear(hidden_size, 64)
        self.linear_lie_prob = nn.Linear(1, 64)
        self.output = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, dense, sparse, length, arc) -> Tensor:
        self.rnn.flatten_parameters()
        if sparse is not None:
            sparse = torch.cat([emb(sparse[..., i]) for i, emb in enumerate(self.sparse_emb)], dim=-1)
            if dense is not None:
                input_ = torch.cat([dense, sparse], dim=-1)
            else:
                input_ = sparse
        else:
            input_ = dense
        input_ = torch.cat([input_, arc[:, None].repeat((1, input_.size(1), 1))], dim=-1)
        input_ = pack_padded_sequence(input_, length.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(input_)

        # output = self.output(h[-1])
        arc = self.linear_lie_prob(arc)
        output = self.linear_h(h[-1])
        output = self.output(F.relu(torch.cat([arc, output], dim=-1)))
        return output[:, 0]
