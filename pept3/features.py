import os

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import similarity, utils
from .dataset import SemiDataset_nfold
from .utils import get_logger


def generate_prosit_feature_set(
    models,
    input_table,
    id2selects,
    save2,
    batch_size=2048,
    gpu_index=0,
    which_sim='SA',
    tensor_need=False,
    tensor_path=None,
    join_unused=True,
):
    utils.set_seed(2022)
    logger = get_logger('features')
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_index}')
    else:
        device = torch.device('cpu')
    logger.info(f'Run on {device}')

    dataset_manager = SemiDataset_nfold(input_table)

    with torch.no_grad():
        sas = []
        sas_tensor = []
        ids_order = []
        for now_i, (model, ids) in enumerate(zip(models, id2selects)):
            logger.info(f'Infer subset [{now_i+1}/{len(models)}]')
            subset_loader = DataLoader(
                dataset_manager.index_data_by_specid(ids),
                batch_size=batch_size,
                shuffle=False,
            )
            t_score, t_tensor = similarity.get_similarity_score_tensor(
                model, subset_loader, which_sim, device=device
            )
            sas.append(t_score)
            sas_tensor.append(t_tensor)
            ids_order.append(np.array(ids))

    scores = np.concatenate(sas)
    score_tensor = np.concatenate(sas_tensor, axis=0)
    Fs = {}
    Fs['SpecId'] = np.concatenate(ids_order)
    Fs[which_sim] = scores

    table_data, frag_msms = dataset_manager.index_dataset_by_specid(Fs['SpecId'])
    Fs['ScanNr'] = np.array(table_data['ScanNr'])
    Fs['Charge'] = np.array(table_data['Charge'])
    Fs['sequence_length'] = np.array(table_data['sequence_length'])
    Fs['collision_energy_aligned_normed'] = np.array(
        table_data['collision_energy_aligned_normed']
    )
    Fs['Peptide'] = np.array(table_data['Peptide'].apply(lambda x: '_.' + x + '._'))
    Fs['Protein'] = np.array(table_data['Peptide'])
    Fs['Label'] = np.array(table_data['Label'])

    pack = [
        (None, None, sa, st, None, frag)
        for sa, st, frag in zip(scores, score_tensor, frag_msms)
    ]

    def add_pred(pack):
        logger.info('Add prosit main features')

        def b(tensor):
            return tensor.reshape(29, 2, 3)[:, 1, :]

        def y(tensor):
            return tensor.reshape(29, 2, 3)[:, 0, :]

        Fs['not_pred_seen'] = [np.sum(m[3][m[5] > 0] == 0) for m in pack]
        Fs['not_pred_seen_b'] = [np.sum(b(m[3])[b(m[5]) > 0] == 0) for m in pack]
        Fs['not_pred_seen_y'] = [np.sum(y(m[3])[y(m[5]) > 0] == 0) for m in pack]
        Fs['pred_nonZero_fragments'] = [np.sum(m[3] > 0) for m in pack]
        Fs['pred_nonZero_b'] = [np.sum(b(m[3]) > 0) for m in pack]
        Fs['pred_nonZero_y'] = [np.sum(y(m[3]) > 0) for m in pack]
        Fs['pred_not_seen'] = [np.sum(m[5][m[3] > 0] == 0) for m in pack]
        Fs['pred_not_seen_b'] = [np.sum(b(m[5])[b(m[3]) > 0] == 0) for m in pack]
        Fs['pred_not_seen_y'] = [np.sum(y(m[5])[y(m[3]) > 0] == 0) for m in pack]
        Fs['pred_seen_nonzero'] = [np.sum(m[5][m[3] > 0] > 0) for m in pack]
        Fs['pred_seen_nonzero_y'] = [np.sum(y(m[5])[y(m[3]) > 0] > 0) for m in pack]
        Fs['pred_seen_nonzero_b'] = [np.sum(b(m[5])[b(m[3]) > 0] > 0) for m in pack]
        Fs['pred_seen_zero'] = [np.sum(m[5][m[3] == 0] == 0) for m in pack]
        Fs['pred_seen_zero_b'] = [np.sum(b(m[5])[b(m[3]) == 0] == 0) for m in pack]
        Fs['pred_seen_zero_y'] = [np.sum(y(m[5])[y(m[3]) == 0] == 0) for m in pack]
        Fs['raw_nonZero_fragments'] = [np.sum(m[5] > 0) for m in pack]
        Fs['raw_nonZero_b'] = [np.sum(b(m[5]) > 0) for m in pack]
        Fs['raw_nonZero_y'] = [np.sum(y(m[5]) > 0) for m in pack]

        theoretically = Fs['sequence_length'] * 2 * Fs['Charge'] + 1e-9
        Fs['rel_not_pred_seen'] = np.array(Fs['not_pred_seen']) / theoretically
        Fs['rel_not_pred_seen_b'] = np.array(Fs['not_pred_seen_b']) / theoretically * 2
        Fs['rel_not_pred_seen_y'] = np.array(Fs['not_pred_seen_y']) / theoretically * 2
        Fs['rel_pred_nonZero_b'] = np.array(Fs['pred_nonZero_b']) / theoretically * 2
        Fs['rel_pred_nonZero_y'] = np.array(Fs['pred_nonZero_y']) / theoretically * 2
        Fs['rel_pred_not_seen'] = np.array(Fs['pred_not_seen']) / theoretically
        Fs['rel_pred_not_seen_b'] = np.array(Fs['pred_not_seen_b']) / theoretically * 2
        Fs['rel_pred_not_seen_y'] = np.array(Fs['pred_not_seen_y']) / theoretically * 2
        Fs['rel_pred_seen_nonzero'] = np.array(Fs['pred_seen_nonzero']) / theoretically
        Fs['rel_pred_seen_nonzero_b'] = (
            np.array(Fs['pred_seen_nonzero_b']) / theoretically * 2
        )
        Fs['rel_pred_seen_nonzero_y'] = (
            np.array(Fs['pred_seen_nonzero_y']) / theoretically * 2
        )
        Fs['rel_pred_seen_zero'] = np.array(Fs['pred_seen_zero']) / theoretically
        Fs['rel_pred_seen_zero_b'] = (
            np.array(Fs['pred_seen_zero_b']) / theoretically * 2
        )
        Fs['rel_pred_seen_zero_y'] = (
            np.array(Fs['pred_seen_zero_y']) / theoretically * 2
        )
        Fs['rel_raw_nonZero_fragments'] = (
            np.array(Fs['raw_nonZero_fragments']) / theoretically
        )
        Fs['rel_raw_nonZero_b'] = np.array(Fs['raw_nonZero_b']) / theoretically * 2
        Fs['rel_raw_nonZero_y'] = np.array(Fs['raw_nonZero_y']) / theoretically * 2

        Fs['relpred_not_pred_seen2pred_nonZero_fragments'] = np.array(
            Fs['not_pred_seen']
        ) / (np.array(Fs['pred_nonZero_fragments']) + 1e-9)
        Fs['relpred_not_pred_seen_b2pred_nonZero_b'] = np.array(
            Fs['not_pred_seen_b']
        ) / (np.array(Fs['pred_nonZero_b']) + 1e-9)
        Fs['relpred_not_pred_seen_y2pred_nonZero_y'] = np.array(
            Fs['not_pred_seen_y']
        ) / (np.array(Fs['pred_nonZero_y']) + 1e-9)
        Fs['relpred_pred_not_seen_b2pred_nonZero_b'] = np.array(
            Fs['pred_not_seen_b']
        ) / (np.array(Fs['pred_nonZero_b']) + 1e-9)
        Fs['relpred_pred_not_seen_y2pred_nonZero_y'] = np.array(
            Fs['pred_not_seen_y']
        ) / (np.array(Fs['pred_nonZero_y']) + 1e-9)

        Fs['relpred_pred_not_seen2pred_nonZero_fragments'] = np.array(
            Fs['pred_not_seen']
        ) / (np.array(Fs['pred_nonZero_fragments']) + 1e-9)
        Fs['relpred_pred_seen_nonzero_b2pred_nonZero_b'] = np.array(
            Fs['pred_seen_nonzero_b']
        ) / (np.array(Fs['pred_nonZero_b']) + 1e-9)
        Fs['relpred_pred_seen_nonzero_y2pred_nonZero_y'] = np.array(
            Fs['pred_seen_nonzero_y']
        ) / (np.array(Fs['pred_nonZero_y']) + 1e-9)

        Fs['relpred_pred_seen_nonzero2pred_nonZero_fragments'] = np.array(
            Fs['pred_seen_nonzero']
        ) / (np.array(Fs['pred_nonZero_fragments']) + 1e-9)
        Fs['relpred_pred_seen_zero_b2pred_nonZero_b'] = np.array(
            Fs['pred_seen_zero_b']
        ) / (np.array(Fs['pred_nonZero_b']) + 1e-9)
        Fs['relpred_pred_seen_zero_y2pred_nonZero_y'] = np.array(
            Fs['pred_seen_zero_y']
        ) / (np.array(Fs['pred_nonZero_y']) + 1e-9)

        Fs['relpred_pred_seen_zero2pred_nonZero_fragments'] = np.array(
            Fs['pred_seen_zero']
        ) / (np.array(Fs['pred_nonZero_fragments']) + 1e-9)

    def add_charge(max_charge=6):
        logger.info(f'Add charge feat, up to {max_charge}')
        for i in range(max_charge):
            Fs[f'Charge{i+1}'] = (Fs['Charge'] == i + 1).astype('int')

    def add_kr():
        Fs['KR'] = np.array(
            table_data['Peptide'].apply(
                lambda x: sum(map(lambda y: 1 if y in 'KR' else 0, x))
            )
        )

    def join_unused_feature():
        unused_name = []
        unwanted_name = ['peak_ions', 'peak_inten', '_numeric_id']
        for key in table_data.keys():
            if key not in Fs and key not in unwanted_name:
                Fs[key] = np.array(table_data[key])
                unused_name.append(key)
        return unused_name

    add_pred(pack)
    add_charge()
    add_kr()

    wanted_features = [
        'SpecId',
        'Label',
        'ScanNr',
        'sequence_length',
        which_sim,
        'KR',
        'collision_energy_aligned_normed',
    ]

    if join_unused:
        unused_name = join_unused_feature()
        logger.info(f'Join unused feature {unused_name}')
        wanted_features = wanted_features + unused_name

    prosit_manual_feat = 'raw_nonZero_fragments  raw_nonZero_y   raw_nonZero_b   pred_nonZero_fragments  pred_nonZero_y  pred_nonZero_b  pred_not_seen  pred_not_seen_y pred_not_seen_b pred_seen_zero  pred_seen_zero_y        pred_seen_zero_b      pred_seen_nonzero        pred_seen_nonzero_y     pred_seen_nonzero_b     not_pred_seen   not_pred_seen_y not_pred_seen_b rel_pred_nonZero_y      rel_pred_nonZero_b      rel_pred_not_seen       rel_pred_not_seen_y    rel_pred_not_seen_b     rel_pred_seen_zero      rel_pred_seen_zero_y    rel_pred_seen_zero_b  rel_pred_seen_nonzero    rel_pred_seen_nonzero_y rel_pred_seen_nonzero_b rel_not_pred_seen       rel_not_pred_seen_y    rel_not_pred_seen_b     relpred_pred_not_seen2pred_nonZero_fragments    relpred_pred_not_seen_y2pred_nonZero_y relpred_pred_not_seen_b2pred_nonZero_b  relpred_pred_seen_zero2pred_nonZero_fragments  relpred_pred_seen_zero_y2pred_nonZero_y relpred_pred_seen_zero_b2pred_nonZero_b relpred_pred_seen_nonzero2pred_nonZero_fragments       relpred_pred_seen_nonzero_y2pred_nonZero_y      relpred_pred_seen_nonzero_b2pred_nonZero_b     relpred_not_pred_seen2pred_nonZero_fragments    relpred_not_pred_seen_y2pred_nonZero_y relpred_not_pred_seen_b2pred_nonZero_b  rel_raw_nonZero_b       rel_raw_nonZero_y     rel_raw_nonZero_fragments'.split()
    wanted_features = (
        wanted_features
        + prosit_manual_feat
        + [
            'Charge1',
            'Charge2',
            'Charge3',
            'Charge4',
            'Charge5',
            'Charge6',
            'Peptide',
            'Protein',
        ]
    )
    output_feature = pd.DataFrame(Fs)[wanted_features]
    if not os.path.exists(os.path.dirname(save2)):
        logger.warning(
            f'Output Path {os.path.dirname(save2)} is not existed, creating it'
        )
        os.mkdir(os.path.dirname(save2))
    output_feature.to_csv(save2, sep='\t', index=False)
    logger.info(f'Saving features to {save2}')
    if tensor_need and tensor_path is not None:
        logger.info(f'Saving tuned tensors into {tensor_path}')
        if not os.path.exists(os.path.dirname(tensor_path)):
            logger.warning(
                f'Output Tensor Path {os.path.dirname(save2)} is not existed, creating it'
            )
            os.mkdir(os.path.dirname(tensor_path))
        f = h5py.File(tensor_path, 'w')
        tensor_dest = f.create_dataset(
            'tuned-tensor', score_tensor.shape, dtype=score_tensor.dtype
        )
        tensor_dest[:] = score_tensor[:]
        ids = [str(n).encode('ascii', 'ignore') for n in Fs['SpecId']]
        f.create_dataset('SpecId', (len(ids),), 'S48', ids)
        f.close()
