import argparse
import logging

import torch

from . import utils
from .features import generate_prosit_feature_set
from .finetune import semisupervised_finetune_twofold
from .models import Model_Factories, Model_Weights_Factories
from .similarity import Similarity_Factories


def parse_args():
    parser = argparse.ArgumentParser(
        description='StuMS: Semi-supervised Fine-tuning for Spectrum Prediction'
    )
    parser.add_argument(
        'input_tab',
        metavar='Input Feature Tab',
        type=str,
        nargs=1,
        help='The input file for StuMS',
    )
    parser.add_argument(
        '--iteration',
        type=int,
        default=10,
        help='maximum iteration times for StuMS, lower this number could reduce running time but also may reduce the performance',
    )
    parser.add_argument(
        '--gpu_index',
        type=int,
        default=0,
        help='GPU id for StuMS, if CUDA is available',
    )
    parser.add_argument(
        '--spmodel',
        type=str,
        default='prosit',
        choices=list(Model_Factories.keys()),
        help='Spectrum Prediction Models, all the models are re-implemented in PyTorch.',
    )
    parser.add_argument(
        '--similarity',
        type=str,
        default='SA',
        choices=list(Similarity_Factories.keys()),
        help='Spectrum Similarity: SA for Spectral Angle, PCC for Pearson Correlation Coefficients',
    )
    parser.add_argument(
        '--loglevel',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error'],
        help='Logging level',
    )
    parser.add_argument(
        '--output_tab',
        type=str,
        default='./prosit_feature.tab',
        help='Output tab file for the fine-tuned prosit features set',
    )
    parser.add_argument(
        '--need_tensor',
        action='store_const',
        const=True,
        default=False,
        help='Need tensor hdf5 file for the fine-tuned spectrum prediction, default False',
    )
    parser.add_argument(
        '--output_tensor',
        type=str,
        default='./stu_tensor.hdf5',
        help='Output tensor hdf5 file for the fine-tuned spectrum prediction',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    utils.LOGGING_LEVEL = getattr(logging, args.loglevel.upper())

    logger = utils.get_logger('MAIN')
    logger.info(f'Set Log Level to {args.loglevel}')
    spectrum_model = Model_Factories[args.spmodel]()
    logger.info(f'Model using: {args.spmodel}')

    table_file = args.input_tab[0]
    logger.info(f'Reading input from {table_file}')

    checkpoint = Model_Weights_Factories(args.spmodel)
    logger.info(f'Load model weights from {checkpoint}')
    spectrum_model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    spectrum_model = spectrum_model.eval()

    fmodel1, fmodel2, id2remove = semisupervised_finetune_twofold(
        spectrum_model,
        table_file,
        gpu_index=args.gpu_index,
        spectrum_sim=args.similarity,
        max_epochs=args.iteration,
    )

    generate_prosit_feature_set(
        fmodel1,
        fmodel2,
        table_file,
        id2remove,
        save2=args.output_tab,
        tensor_need=args.need_tensor,
        tensor_path=args.output_tensor,
    )
