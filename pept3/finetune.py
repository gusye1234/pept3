import sys
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import similarity, utils
from .dataset import SemiDataset_nfold
from .utils import get_logger


def pept3_nfold_finetune(
    ori_model,
    input_table,
    batch_size=2048,
    gpu_index=0,
    nfold=3,
    max_epochs=10,
    update_interval=1,
    q_threshold=0.1,
    validate_q_threshold=0.01,
    spectrum_sim='SA',
    enable_test=False,
    only_id2select=False,
):
    utils.set_seed(2022)
    logger = get_logger('finetune')
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_index}')
    else:
        device = torch.device('cpu')
        logger.warning(
            'No Cuda is detected. Use CPU now, which can slow down the computing'
        )
    logger.debug(f'Run on {device}')

    def finetune(dataset: SemiDataset_nfold):
        model = deepcopy(ori_model)
        model = model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, eps=1e-8)
        loss_fn = similarity.FinetuneSALoss(spectrum_sim=spectrum_sim)
        model = model.to(device)
        data_loader = DataLoader(
            dataset.train_all_data(), batch_size=batch_size, shuffle=False
        )
        logger.info(
            f'max iteration {max_epochs}, watching FDR threshold: {validate_q_threshold}, use --loglevel=debug to see the details in training'
        )
        if dataset._scores is not None:
            q_values = dataset.Q_values()
            logger.info(
                f'Baseline: FDR@[0.001, 0.01, 0.1]: {(np.sum(q_values < 0.001), np.sum(q_values < 0.01), np.sum(q_values < 0.1))}'
            )
        best_model = None
        best_q_value_num = 0
        for epoch in range(max_epochs):
            loss = 0
            loss_l1 = 0.0
            loss_sa = 0.0
            if (epoch % update_interval) == 0:
                with torch.no_grad():
                    scores = similarity.get_similarity_score(
                        model, data_loader, spectrum_sim, device=device
                    )
                    dataset.assign_train_score(scores)
                    q_values = dataset.Q_values()
                    q_values_num = np.sum(q_values < validate_q_threshold)
                    if epoch == 0:
                        logger.info(
                            f'{spectrum_sim} for FDR@[0.001, 0.01, 0.1]: {(np.sum(q_values < 0.001), np.sum(q_values < 0.01), np.sum(q_values < 0.1))}'
                        )
                if q_values_num > best_q_value_num:
                    best_model = deepcopy(model)
                    best_q_value_num = q_values_num
                    logger.debug(
                        f'FDR@{validate_q_threshold}: {np.sum(q_values < 0.01)}*'
                    )
                else:
                    logger.debug(
                        f'FDR@{validate_q_threshold}: {np.sum(q_values < 0.01)}'
                    )
                train_loader = DataLoader(
                    dataset.semisupervised_sa_finetune(threshold=q_threshold),
                    batch_size=batch_size,
                    shuffle=True,
                )
            model = model.train()
            for i, data in enumerate(train_loader):
                data = {k: v.to(device) for k, v in data.items()}
                data['peptide_mask'] = utils.create_mask(data['sequence_integer'])
                pred = model(data)
                loss_b, fine_loss, l1_loss = loss_fn(
                    data['intensities_raw'], pred, data['label']
                )
                optimizer.zero_grad()
                loss_b.backward()
                optimizer.step()
                loss += loss_b.item()
                loss_l1 += l1_loss
                loss_sa += fine_loss
        with torch.no_grad():
            scores = similarity.get_similarity_score(
                best_model, data_loader, spectrum_sim, device=device
            )
            dataset.assign_train_score(scores)
            q_values = dataset.Q_values()
            logger.info(
                f'{spectrum_sim} with PepT3 for FDR@[0.001, 0.01, 0.1]: {(np.sum(q_values < 0.001), np.sum(q_values < 0.01), np.sum(q_values < 0.1))}'
            )
        return best_model

    dataset_manager = SemiDataset_nfold(input_table, nfold=nfold)
    id2select = dataset_manager.id2predict()
    if only_id2select:
        return [ori_model for _ in range(nfold)], id2select
    models = []
    for i in range(nfold):
        dataset_manager.set_index(i)
        logger.info(
            f'Training ({i+1}/{nfold})... train set {len(dataset_manager._d)}, test set {len(dataset_manager._test_d)}'
        )
        model = finetune(dataset_manager)
        models.append(model)
    return models, id2select
