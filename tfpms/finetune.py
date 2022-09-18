import sys
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import similarity, utils
from .dataset import SemiDataset
from .utils import get_logger


def semisupervised_finetune_twofold(
    ori_model,
    input_table,
    batch_size=2048,
    gpu_index=0,
    max_epochs=10,
    update_interval=1,
    q_threshold=0.1,
    validate_q_threshold=0.01,
    spectrum_sim='SA',
    enable_test=False,
    only_id2remove=False,
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

    def finetune(dataset: SemiDataset):
        model = deepcopy(ori_model)
        model = model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, eps=1e-8)
        loss_fn = similarity.FinetuneSALoss(spectrum_sim=spectrum_sim)
        model = model.to(device)
        data_loader = DataLoader(
            dataset.train_all_data(), batch_size=batch_size, shuffle=False
        )
        if dataset._scores is not None:
            q_values = dataset.Q_values()
            logger.info(
                f'Baseline Score for FDR@[0.001, 0.01, 0.1]: {(np.sum(q_values < 0.001), np.sum(q_values < 0.01), np.sum(q_values < 0.1))}'
            )
            logger.info('---------------')
        logger.info(
            f'max iteration {max_epochs}, watching FDR threshold: {validate_q_threshold}, use --loglevel=debug to see the details in training'
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
                            f'No fine-tuned {spectrum_sim} for FDR@[0.001, 0.01, 0.1]: {(np.sum(q_values < 0.001), np.sum(q_values < 0.01), np.sum(q_values < 0.1))}'
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
                f'   Finetuned {spectrum_sim} for FDR@[0.001, 0.01, 0.1]: {(np.sum(q_values < 0.001), np.sum(q_values < 0.01), np.sum(q_values < 0.1))}'
            )
        return best_model

    dataset_manager = SemiDataset(input_table)
    id2remove = dataset_manager.id2remove()  # default first part
    if only_id2remove:
        return ori_model, ori_model, id2remove
    logger.info('Training (1/2)...')
    logger.info('---------------')
    model1 = finetune(dataset_manager)

    dataset_manager = dataset_manager.reverse()
    logger.info('Training (2/2)...')
    logger.info('---------------')
    model2 = finetune(dataset_manager)
    return model1, model2, id2remove
