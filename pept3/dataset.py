import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .bio import one_hot, peptide_to_inter, reverse_annotation
from .utils import get_logger


def Debug(msg):
    logger = get_logger('dataset')
    logger.debug(msg)


def Info(msg):
    logger = get_logger('dataset')
    logger.info(msg)


def Error(msg):
    logger = get_logger('dataset')
    logger.error(msg)


class SemiDataset:
    def __init__(self, table_input, score_init='andromeda', pi=0.9):
        self._file_input = table_input
        self._pi = pi
        if table_input.endswith('tab'):
            self._hdf5 = False
            self._data = (
                pd.read_csv(table_input, sep='\t')
                .sample(frac=1, random_state=2022)
                .reset_index(drop=True)
            )
            self._data['sequence_length'] = self._data['Peptide'].apply(
                lambda x: len(x)
            )
            self._frag_msms = self.backbone_spectrums()

        elif table_input.endswith('hdf5'):
            Error('Hdf5 input currently no enabled')
            raise NotImplementedError('Hdf5 input currently no enabled')
            # self._hdf5 = True
            # _feat = h5py.File(table_input, 'r')
            # # Peptide, Charge, collision_energy_aligned_normed, Label,
            # _label = np.array(_feat['reverse']).astype("int")
            # _label[_label == 1] = -1
            # _label[_label == 0] = 1
            # self._data = pd.DataFrame({
            #     "Peptide_integer": list(np.array(_feat['sequence_integer'])),
            #     "Charge_onehot": list(np.array(_feat['precursor_charge_onehot'])),
            #     "Label": _label.squeeze(),
            #     "andromeda": np.array(_feat['score']).squeeze(),
            #     "collision_energy_aligned_normed": np.array(_feat['collision_energy_aligned_normed']).squeeze(),
            #     "SpecId": np.arange(len(_label))
            # })
            # self._frag_msms = np.array(_feat['intensities_raw'])
            # order = np.arange(len(self._frag_msms))
            # np.random.shuffle(order)
            # self._data = self._data.reindex(order)
            # self._frag_msms = self._frag_msms[order]
        Info(f'Total {len(self._frag_msms)} data were loaded')
        self._data['_numeric_id'] = np.arange(len(self._data))
        self._d, self._df, self._test_d, self._test_df = self.split_dataset()
        self._scores = None
        self._test_scores = None
        self._baseline_score_name = None
        if score_init is not None:
            self._baseline_score_name = score_init
            self.assign_train_score(self._d[score_init])
            self.assign_test_score(self._test_d[score_init])

    def reverse(self):
        Debug(f'[Dataset reverse]: {len(self._d)} -> {len(self._test_d)}')
        self._d, self._test_d = self._test_d, self._d
        self._df, self._test_df = self._test_df, self._df
        self._scores = None
        self._test_scores = None
        if self._baseline_score_name is not None:
            self.assign_train_score(self._d[self._baseline_score_name])
            self.assign_test_score(self._test_d[self._baseline_score_name])
        return self

    def backbone_spectrums(self):
        sp = self._data.apply(
            lambda x: reverse_annotation(
                x['peak_ions'], x['peak_inten'], x['Charge'], x['sequence_length']
            ).reshape(1, -1),
            axis=1,
        )
        return np.concatenate(sp, axis=0)

    def specid_twofold_division(self, id2remove):
        target_id = self._data[self._data['Label'] == 1]['SpecId'].to_list()
        decoy1_id = id2remove
        decoy_id = self._data[self._data['Label'] == -1]['SpecId'].to_list()
        decoy2_id = [n for n in decoy_id if n not in id2remove]

        return np.array(target_id), np.array(decoy1_id), np.array(decoy2_id)

    def assign_train_score(self, scores):
        assert len(scores) == len(self._d)
        self._scores = np.array(scores)

    def assign_test_score(self, scores):
        assert len(scores) == (len(self._test_d))
        self._test_scores = np.array(scores)

    def split_dataset(self):
        targets = self._data[self._data['Label'] == 1]
        targets_frag = self._frag_msms[self._data['Label'] == 1]
        decoys = self._data[self._data['Label'] == -1]
        decoys_frag = self._frag_msms[self._data['Label'] == -1]

        train_data = pd.concat([targets, decoys[: len(decoys) // 2]])
        train_frag = np.concatenate(
            (targets_frag, decoys_frag[: len(decoys) // 2]), axis=0
        )
        test_data = pd.concat([targets, decoys[len(decoys) // 2 :]])
        test_decoy_frag = np.concatenate(
            (targets_frag, decoys_frag[len(decoys) // 2 :]), axis=0
        )
        return train_data, train_frag, test_data, test_decoy_frag

    def id2remove(self):
        return np.array(self._d[self._d['Label'] == -1]['SpecId'])

    def q_compute(self, scores, table, pi):
        ratio = (table['Label'] == 1).sum() / (table['Label'] == -1).sum()
        ratio = pi * ratio
        indexs = np.arange(len(scores))
        labels = np.array(table['Label'])
        orders = np.argsort(scores)

        indexs = indexs[orders]
        labels = labels[orders]

        target_sum = np.flip(np.cumsum(np.flip(labels == 1)))
        decoy_sum = np.flip(np.cumsum(np.flip(labels == -1)))

        target_sum[:-1] = target_sum[1:]
        decoy_sum[:-1] = decoy_sum[1:]

        fdrs = ratio * decoy_sum / (target_sum + 1e-9)
        fdrs[-1] = 0
        q_values = np.zeros_like(fdrs)
        min_fdrs = np.inf
        for i, fdr in enumerate(fdrs):
            min_fdrs = min(min_fdrs, fdr)
            q_values[i] = min_fdrs

        remap = np.argsort(indexs)
        q_values = q_values[remap]
        return q_values

    def Q_values(self):
        q_values = self.q_compute(self._scores, self._d, self._pi)
        return q_values

    def Q_values_test(self):
        q_values = self.q_compute(self._test_scores, self._test_d, self._pi)
        return q_values

    def prepare_sa_data(self, table, frag_msms):
        xlabel = [
            'sequence_integer',
            'precursor_charge_onehot',
            'collision_energy_aligned_normed',
        ]
        ylabel = 'intensities_raw'
        names = xlabel + [ylabel, 'label']

        y_data = torch.from_numpy(frag_msms)
        if not self._hdf5:
            seq_data = list(
                table.apply(lambda x: peptide_to_inter(x['Peptide']), axis=1)
            )
            seq_data = torch.from_numpy(np.concatenate(seq_data))
        else:
            seq_data = [i.reshape(1, -1) for i in table['Peptide_integer'].to_list()]
            seq_data = torch.from_numpy(np.concatenate(seq_data))

        if not self._hdf5:
            charges = list(table.apply(lambda x: one_hot(x['Charge'] - 1), axis=1))
            charges = torch.from_numpy(np.concatenate(charges))
        else:
            charges = [i.reshape(1, -1) for i in table['Charge_onehot'].to_list()]
            charges = torch.from_numpy(np.concatenate(charges))

        nces = np.array(table['collision_energy_aligned_normed'])
        nces = torch.from_numpy(nces).unsqueeze(1)

        labels = np.array(table['Label'])
        labels = torch.from_numpy(labels)

        data_sa = [seq_data, charges, nces, y_data, labels]
        return names, data_sa

    def semisupervised_sa_finetune(self, threshold=0.1):
        q_values = self.q_compute(self._scores, self._d, self._pi)
        sat_d = self._d[q_values <= threshold]
        sat_f = self._df[q_values <= threshold]
        names, data_sa = self.prepare_sa_data(sat_d, sat_f)
        return FinetuneTableDataset(names, data_sa)

    def supervised_sa_finetune(self):
        return self.train_all_data()

    def train_all_data(self):
        names, data_sa = self.prepare_sa_data(self._d, self._df)
        return FinetuneTableDataset(names, data_sa)

    def test_all_data(self):
        names, data_sa = self.prepare_sa_data(self._test_d, self._test_df)
        return FinetuneTableDataset(names, data_sa)

    def all_data(self):
        names, data_sa = self.prepare_sa_data(self._data, self._frag_msms)
        return FinetuneTableDataset(names, data_sa)

    def index_dataset_by_specid(self, specid):
        assert self._data['SpecId'].is_unique
        spec_data = self._data.set_index('SpecId')
        wanted_table = spec_data.loc[specid]
        wanted_frag = self._frag_msms[wanted_table['_numeric_id'].to_numpy()]
        return wanted_table, wanted_frag

    def index_data_by_specid(self, specid):
        wanted_table, wanted_frag = self.index_dataset_by_specid(specid)
        names, data_sa = self.prepare_sa_data(wanted_table, wanted_frag)
        return FinetuneTableDataset(names, data_sa)


class SemiDataset_nfold(SemiDataset):
    def __init__(
        self, table_input, nfold=3, score_init='andromeda', pi=0.9, rawfile_fiels=None
    ):
        self._file_input = table_input
        self._pi = pi
        if not table_input.endswith('hdf5'):
            self._hdf5 = False
            self._data = (
                pd.read_csv(table_input, sep='\t')
                .sample(frac=1, random_state=2022)
                .reset_index(drop=True)
            )
            self._frag_msms = self.backbone_spectrums()
        else:
            self._hdf5 = True
            Error('Hdf5 input currently no enabled')
            raise NotImplementedError('Hdf5 input currently no enabled')
            # if rawfile_fiels is None:
            #     _feat = h5py.File(table_input, 'r')
            # else:
            #     pass
            # # Peptide, Charge, collision_energy_aligned_normed, Label,
            # _label = np.array(_feat['reverse']).astype("int")
            # _label[_label == 1] = -1
            # _label[_label == 0] = 1
            # self._data = pd.DataFrame({
            #     "Peptide_integer": list(np.array(_feat['sequence_integer'])),
            #     "Charge_onehot": list(np.array(_feat['precursor_charge_onehot'])),
            #     "Label": _label.squeeze(),
            #     "andromeda": np.array(_feat['score']).squeeze(),
            #     "collision_energy_aligned_normed": np.array(_feat['collision_energy_aligned_normed']).squeeze()
            # })
            # self._frag_msms = np.array(_feat['intensities_raw'])
            # print(f"Total {len(self._frag_msms)} data loader from hdf5")
        Info(f'Total {len(self._frag_msms)} data were loaded')
        self._data['_numeric_id'] = np.arange(len(self._data))
        self._nfold = nfold
        self._score_init = score_init
        self._nfold_index = self.split_dataset()
        self.set_index(0)

    def set_index(self, index):
        assert index < self._nfold
        self._index = index
        self._d, self._df = self.index2dataset(*self._nfold_index[index][:2])
        self._test_d, self._test_df = self.index2dataset(*self._nfold_index[index][2:])
        self.assign_train_score(self._d[self._score_init])
        self.assign_test_score(self._test_d[self._score_init])
        return self

    def index2dataset(self, target_index, decoy_index):
        # ms_data = self._data.iloc[target_index].append(self._data.iloc[decoy_index])
        ms_data = pd.concat(
            [self._data.iloc[target_index], self._data.iloc[decoy_index]],
            ignore_index=True,
        )
        ms_frag = np.concatenate(
            (self._frag_msms[target_index], self._frag_msms[decoy_index]), axis=0
        )
        return ms_data, ms_frag

    def select_ids(self, id2select):
        pass

    def split_dataset(self):
        targets_index = self._data[self._data['Label'] == 1].index.values
        decoys_index = self._data[self._data['Label'] == -1].index.values

        len_test_target = int(len(targets_index) / self._nfold)
        len_test_decoy = int(len(decoys_index) / self._nfold)
        nfold_index = []
        for i in range(self._nfold):
            t_start = i * len_test_target
            t_end = (
                (i + 1) * len_test_target
                if i != (self._nfold - 1)
                else len(targets_index)
            )
            d_start = i * len_test_decoy
            d_end = (
                (i + 1) * len_test_decoy
                if i != (self._nfold - 1)
                else len(decoys_index)
            )

            test_target = targets_index[t_start:t_end]
            test_decoy = decoys_index[d_start:d_end]

            train_target = np.concatenate(
                [targets_index[:t_start], targets_index[t_end:]]
            )
            train_decoy = np.concatenate([decoys_index[:d_start], decoys_index[d_end:]])
            nfold_index.append((train_target, train_decoy, test_target, test_decoy))

        return nfold_index

    def id2predict(self):
        predictable_ids = []
        if not self._hdf5:
            for i in range(self._nfold):
                test_d, _ = self.index2dataset(*self._nfold_index[i][2:])
                predictable_ids.append(test_d['SpecId'])
        else:
            for i in range(self._nfold):
                test_d, _ = self.index2dataset(*self._nfold_index[i][2:])
                predictable_ids.append(test_d.index.values)
        return predictable_ids


class pDeep_nfold(SemiDataset_nfold):
    def pdeep_train_score(self):
        total_len = len(self._df)
        frag_msms = self._df.reshape(total_len, 29, 2, 3)

        pep_len = self._d['sequence_length'].to_list()
        pep_len = np.array(pep_len)

        b_ions = frag_msms[:, :, 0, :].reshape(total_len, -1)
        y_ions = frag_msms[:, :, 1, :].reshape(total_len, -1)

        scores = []
        for i in range(total_len):
            bion = b_ions[i]
            yion = y_ions[i]
            if self._d.iloc[i]['Label'] == -1:
                scores.append(-float('inf'))
                continue
            s1 = np.log(np.sum(bion[bion > 0]) * np.sum(bion > 0) / pep_len[i] + 1e-7)
            s2 = np.log(np.sum(yion[yion > 0]) * np.sum(yion > 0) / pep_len[i] + 1e-7)
            scores.append(s1 + s2)
        return scores

    def pdeep3_finetune(self, max_sample=100):
        max_sample_index = np.argsort(self._scores)[-max_sample:]
        sat_d = self._d.iloc[max_sample_index]
        sat_f = self._df[max_sample_index]
        assert all(
            sat_d['Label'].apply(lambda x: x == 1)
        ), f'Target PSMs is lower than {max_sample}'
        names, data_sa = self.prepare_sa_data(sat_d, sat_f)
        return FinetuneTableDataset(names, data_sa)


class FinetuneTableDataset(Dataset):
    def __init__(self, names, xs):
        self.x = xs
        self.names = names

    def __len__(self):
        return len(self.x[0])

    def __getitem__(self, index):
        re = [i[index] for i in self.x]
        return {self.names[i]: re[i] for i in range(len(re))}
