import torch
import pytorch_lightning as pl
import sys
import warnings
import os

from argparse import ArgumentParser
from tools.train import train
from tools.test import test
from config import TRAIN_DIRS, TEST_DIRS, VAL_DIRS


def parse_arguments(argv):
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)

    # Model hyperparameters
    parser.add_argument('--model', type=str, default='resnet18', help='Model name')
    parser.add_argument('--mlp_structures', type=int, nargs='+', default=[2, ], help='MLP structures')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='Dropout rate')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for feature extractor")
    parser.add_argument('--lr_head', type=float, default=1e-3, help="Learning rate for classifier head")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs")
    parser.add_argument('--clip_grad', type=float, default=5.0, help="Gradient clipping value")

    parser.add_argument('--train_full', action='store_true', help="Train on full dataset")

    return parser.parse_args(argv)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = parse_arguments(sys.argv[1:])

    # Set seed
    pl.seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Set data directories
    args.train_dirs = TRAIN_DIRS
    args.val_dirs = VAL_DIRS
    args.test_dirs = TEST_DIRS

    # Train model
    model = train(args)

    # Test model
    test(model, args)