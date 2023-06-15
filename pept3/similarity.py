import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .utils import *


def spectral_angle(true, pred):
    true_mask = (true >= 0).float()

    pred2com = pred * true_mask
    true2com = true * true_mask

    pred2com = F.normalize(pred2com)
    true2com = F.normalize(true2com)

    re = torch.sum(pred2com * true2com, dim=-1)
    re[re > 1] = 1
    re[re < -1] = -1
    return 1 - (2 / torch.pi) * torch.arccos(re)


def pearson_coff(true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Computes the Pearson correlation coefficient between the true and predicted tensors.

    Args:
        true (torch.Tensor): The true tensor.
        pred (torch.Tensor): The predicted tensor.

    Returns:
        torch.Tensor: The Pearson correlation coefficient between the true and predicted tensors.
    """
    # Create a mask to zero out negative values in the true tensor
    true_mask = (true >= 0).float()

    # Apply the mask to the predicted tensor
    pred2com = pred * true_mask
    true2com = true * true_mask

    # Subtract the mean of each row from the corresponding row in the predicted and true tensors
    pred2com -= torch.mean(pred2com, dim=1).unsqueeze(-1)
    true2com -= torch.mean(true2com, dim=1).unsqueeze(-1)

    # Normalize the rows of the predicted and true tensors
    pred2com = F.normalize(pred2com, dim=1)
    true2com = F.normalize(true2com, dim=1)

    # Compute the dot product between each row of the predicted and true tensors and sum them up
    return torch.sum(pred2com * true2com, dim=-1)


def predict_sa(true, pred):
    true_mask = (true >= 0).float()
    pred = pred / pred.max()
    pred = pred * true_mask
    return spectral_angle(true, pred), pred


def predict_pearson(true, pred):
    true_mask = (true >= 0).float()
    pred = pred / pred.max()
    pred = pred * true_mask
    return pearson_coff(true, pred), pred


Similarity_Factories = {'SA': spectral_angle, 'PCC': pearson_coff}

_Similarity_Factories = {'SA': predict_sa, 'PCC': predict_pearson}


class FinetuneSALoss(nn.Module):
    def __init__(self, l1_lambda=0.000001, spectrum_sim=False):
        super().__init__()
        self.l1_lambda = l1_lambda
        self.sim = spectrum_sim
        if self.sim not in Similarity_Factories:
            raise NotImplementedError(
                f'Spectrum Similarity method {self.sim} is not supported yet.'
            )

    def forward(self, true, pred, label):
        true_mask = (true >= 0).float()
        pred = pred * true_mask
        l1_v = torch.abs(pred).sum(1).mean()
        scores = Similarity_Factories[self.sim](true, pred)
        base = torch.mean((scores - label) ** 2)
        return base + self.l1_lambda * l1_v, base.item(), l1_v.item()


def get_similarity_score(model, data_loader, which_sim, device=torch.device('cpu')):
    with torch.no_grad():
        model = model.eval()
        scores = []
        if which_sim not in Similarity_Factories:
            raise NotImplementedError(
                f'Spectrum Similarity method {which_sim} is not supported yet.'
            )
        for i, data in enumerate(data_loader):
            data = {k: v.to(device) for k, v in data.items()}
            data['peptide_mask'] = create_mask(data['sequence_integer'])
            pred = model(data)
            sas = Similarity_Factories[which_sim](data['intensities_raw'], pred)
            scores.append(sas.detach().cpu().numpy())
        scores = np.concatenate(scores, axis=0)
        return scores


def get_similarity_score_tensor(
    model, data_loader, which_sim, device=torch.device('cpu')
):
    with torch.no_grad():
        model = model.eval()
        scores = []
        pred_tensors = []
        if which_sim not in Similarity_Factories:
            raise NotImplementedError(
                f'Spectrum Similarity method {which_sim} is not supported yet.'
            )
        for i, data in enumerate(data_loader):
            data = {k: v.to(device) for k, v in data.items()}
            data['peptide_mask'] = create_mask(data['sequence_integer'])
            pred = model(data)
            sas, pred = _Similarity_Factories[which_sim](data['intensities_raw'], pred)
            scores.append(sas.detach().cpu().numpy())
            pred_tensors.append(pred.detach().cpu().numpy())
        scores = np.concatenate(scores, axis=0)
        pred_tensors = np.concatenate(pred_tensors, axis=0)
    return scores, pred_tensors
