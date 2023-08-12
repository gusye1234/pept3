import hashlib
import os

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .utils import download_file, get_logger

logger = get_logger('model')


class AttentalSum(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.w = nn.Linear(input_dim, 1)
        self.act = nn.Tanh()
        self.soft = nn.Softmax(dim=0)

    def forward(self, x: Tensor, src_mask: Tensor = None):
        # x: S B D, src_mask: B S
        weight = self.w(x)
        weight = self.act(weight).clone()

        if src_mask is not None:
            weight[src_mask.transpose(0, 1)] = -torch.inf
        weight = self.soft(weight)

        weighted_embed = torch.sum(x * weight, dim=0)
        return weighted_embed


class pDeep2_nomod(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_size = 256
        self.peptide_dim = kwargs.pop('peptide_dim', 22)
        self.instrument_size = 8
        self.input_size = self.peptide_dim * 4 + 2 + 1 + 3
        self.ions_dim = kwargs.pop('ions_dim', 6)
        self.instrument_ce_scope = 'instrument_nce'
        self.rnn_dropout = 0.2
        self.output_dropout = 0.2
        self.init_layers()

    def init_layers(self):
        self.lstm_layer1 = nn.LSTM(
            self.input_size, self.layer_size, batch_first=True, bidirectional=True
        )
        self.lstm_layer2 = nn.LSTM(
            self.layer_size * 2 + 1 + 3,
            self.layer_size,
            batch_first=True,
            bidirectional=True,
        )

        self.lstm_output_layer = nn.LSTM(
            self.layer_size * 2 + 1 + 3,
            self.ions_dim,
            bidirectional=True,
            batch_first=True,
        )
        self.linear_inst_proj = nn.Linear(self.instrument_size + 1, 3, bias=False)
        self.dropout = nn.Dropout(p=self.output_dropout)

    def comment(self):
        return 'pDeep2'

    def pdeep2_long_feature(self, data):
        peptides = F.one_hot(data['sequence_integer'], num_classes=self.peptide_dim)
        peptides_mask = data['peptide_mask']
        peptides_length = torch.sum(peptides_mask, dim=1)
        pep_dim = peptides.shape[2]
        assert pep_dim == self.peptide_dim
        long_feature = peptides.new_zeros(
            (peptides.shape[0], peptides.shape[1] - 1, pep_dim * 4 + 2)
        )
        long_feature[:, :, :pep_dim] = peptides[:, :-1, :]
        long_feature[:, :, pep_dim : 2 * pep_dim] = peptides[:, 1:, :]
        for i in range(peptides.shape[1] - 1):
            long_feature[:, i, 2 * pep_dim : 3 * pep_dim] = (
                torch.sum(peptides[:, :i, :], dim=1)
                if i != 0
                else peptides.new_zeros((peptides.shape[0], pep_dim))
            )
            long_feature[:, i, 3 * pep_dim : 4 * pep_dim] = (
                torch.sum(peptides[:, (i + 2) :, :], dim=1)
                if i == (peptides.shape[1] - 2)
                else peptides.new_zeros((peptides.shape[0], pep_dim))
            )
            long_feature[:, i, 4 * pep_dim] = 1 if (i == 0) else 0
            long_feature[:, i, 4 * pep_dim + 1] = (peptides_length - 2) == i
        return long_feature

    def add_leng_dim(self, x, length):
        x = x.unsqueeze(dim=1)
        shape_repeat = [1] * len(x.shape)
        shape_repeat[1] = length
        return x.repeat(*shape_repeat)

    def forward(self, data, **kwargs):
        peptides = self.pdeep2_long_feature(data)  # n-1 input

        nce = data['collision_energy_aligned_normed'].float()
        charge = data['precursor_charge_onehot'].float()
        charge = torch.argmax(charge, dim=1).unsqueeze(-1)

        B = peptides.shape[0]
        peptides_length = peptides.shape[1]
        inst_feat = charge.new_zeros((B, self.instrument_size))
        # ['QE', 'Velos', 'Elite', 'Fusion', 'Lumos', 'unknown']
        inst_feat[:5] = 1
        charge = self.add_leng_dim(charge, peptides_length)
        nce = self.add_leng_dim(nce, peptides_length)
        inst_feat = self.add_leng_dim(inst_feat, peptides_length)

        proj_inst = self.linear_inst_proj(torch.cat([inst_feat, nce], dim=2))
        x = torch.cat([peptides, charge, proj_inst], dim=2)

        x, _ = self.lstm_layer1(x)
        x = self.dropout(x)
        x = torch.cat([x, charge, proj_inst], dim=2)
        x, _ = self.lstm_layer2(x)
        x = self.dropout(x)
        x = torch.cat([x, charge, proj_inst], dim=2)
        output, _ = self.lstm_output_layer(x)
        output = output[:, :, : self.ions_dim] + output[:, :, self.ions_dim :]

        return output.reshape(B, -1)


class PrositFrag(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.peptide_dim = kwargs.pop('peptide_dim', 22)
        self.peptide_embed_dim = kwargs.pop('peptide_embed_dim', 32)
        self.percursor_dim = kwargs.pop('peptide_embed_dim', 6)
        self.hidden_size = kwargs.pop('bi_dim', 256)
        self.max_sequence = kwargs.pop('max_lenght', 30)

        self.embedding = nn.Embedding(self.peptide_dim, self.peptide_embed_dim)
        self.bi = nn.GRU(
            input_size=self.peptide_embed_dim,
            hidden_size=self.hidden_size,
            bidirectional=True,
        )
        self.drop3 = nn.Dropout(p=0.3)
        self.gru = nn.GRU(
            input_size=self.hidden_size * 2, hidden_size=self.hidden_size * 2
        )
        self.agg = AttentalSum(self.hidden_size * 2)
        self.leaky = nn.LeakyReLU()

        self.side_encoder = nn.Linear(self.percursor_dim + 1, self.hidden_size * 2)

        self.gru_decoder = nn.GRU(
            input_size=self.hidden_size * 2, hidden_size=self.hidden_size * 2
        )
        self.in_frag = nn.Linear(self.max_sequence - 1, self.max_sequence - 1)
        self.final_decoder = nn.Linear(self.hidden_size * 2, 6)

    def comment(self):
        return 'PrositFrag'

    def forward(self, x, **kwargs):
        self.bi.flatten_parameters()
        self.gru.flatten_parameters()
        self.gru_decoder.flatten_parameters()

        peptides = x['sequence_integer']
        nce = x['collision_energy_aligned_normed'].float().reshape(-1, 1)
        charge = x['precursor_charge_onehot'].float()
        B = peptides.shape[0]
        x = self.embedding(peptides.int())
        x = x.transpose(0, 1)
        x, _ = self.bi(x)
        x = self.drop3(x)
        x, _ = self.gru(x)
        x = self.drop3(x)
        x = self.agg(x)

        side_input = torch.cat([charge, nce], dim=1)
        side_info = self.side_encoder(side_input)
        side_info = self.drop3(side_info)

        x = x * side_info
        x = x.expand(self.max_sequence - 1, x.shape[0], x.shape[1])
        x, _ = self.gru_decoder(x)
        x = self.drop3(x)
        x_d = self.in_frag(x.transpose(0, 2))

        x = x * x_d.transpose(0, 2)
        x = self.final_decoder(x)
        x = self.leaky(x)
        x = x.transpose(0, 1).reshape(B, -1)
        return x


_Model_Factories = {'prosit': PrositFrag, 'pdeep': pDeep2_nomod}
ROOT_DIR = os.path.expanduser('~/.pept3_cache')
if not os.path.exists(ROOT_DIR):
    os.mkdir(ROOT_DIR)

_Model_Weights_Factories = {
    'prosit': os.path.join(ROOT_DIR, 'best_frag_l1_PrositFrag-1024.pth'),
    'pdeep': os.path.join(ROOT_DIR, 'best_frag_l1_pDeep2-1024.pth'),
}

_Model_Weights_Url_Factories = {
    'prosit': 'https://github.com/gusye1234/pept3/raw/main/assets/best_frag_l1_PrositFrag-1024.pth',
    'pdeep': 'https://github.com/gusye1234/pept3/raw/main/assets/best_frag_l1_pDeep2-1024.pth',
}

_Model_Weights_md5_Factories = {
    'prosit': 'ec4bc8a7761c38f8732f5f53c3ec40ff',
    'pdeep': 'ae4deb4efbc963c57bd1420ba5df109e',
}


def download_model(model_name, model_path):
    model_url = _Model_Weights_Url_Factories[model_name]
    download_file(model_url, model_path)


def Model_Factories(model_name):
    return _Model_Factories[model_name]()


def Model_Weights_Factories(model_name):
    model_path = _Model_Weights_Factories[model_name]
    if not os.path.exists(model_path):
        print(f'Download {model_name} to {ROOT_DIR}')
        download_model(model_name, model_path)
    else:
        print(f'Load {model_name} from {model_path}')
    with open(model_path, 'rb') as weights_bin:
        md5_test = hashlib.md5()
        md5_test.update(weights_bin.read())
        assert (
            md5_test.hexdigest() == _Model_Weights_md5_Factories[model_name]
        ), f'Wrong md5 checksum, please remove {model_path} and re-run'
        logger.info(f'{model_name} passed checksum')
        return model_path
