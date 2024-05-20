import argparse
import json

import wandb
from data_handler import *  # noqa: F403
from datasets import *  # noqa: F403
from model_handler import *  # noqa: F403
from models import *  # noqa: F403
from utils import *  # noqa: F403


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_config_path', type=str, default='./template.json')
    args = vars(parser.parse_args())
    return args


def main(args) -> None:
    # [STEP-1] Open the configuration file and get arguments.
    with open(args['exp_config_path']) as f:
        args = json.load(f)

    wandb.init(
        # set the wandb project where this run will be logged
        project='drag',
        # track hyperparameters and run metadata
        config={
            'seed': args['seed'],
            'data_name': args['data_name'],
            'multi_relation': args['multi_relation'],
            'n_head': args['n_head'],
            'n_head_agg': args['n_head_agg'],
            'feat_drop': args['feat_drop'],
            'attn_drop': args['attn_drop'],
            'train_ratio': args['train_ratio'],
            'test_ratio': args['test_ratio'],
            'emb_size': args['emb_size'],
            'lr': args['lr'],
            'weight_decay': args['weight_decay'],
            'epochs': args['epochs'],
            'valid_epochs': args['valid_epochs'],
            'batch_size': args['batch_size'],
            'patience': args['patience'],
            'cuda_id': args['cuda_id'],
            'save_dir': args['save_dir'],
        },
    )

    # [STEP-2] Initialize the Datahandler object to handle the fraud detection dataset.
    data_handler = DataHandlerModule(args)  # noqa: F405

    # [STEP-3] Train model and evaluate the performance.
    model = ModelHandlerModule(args, data_handler)  # noqa: F405
    _, _ = model.train()


if __name__ == '__main__':
    args = get_arguments()
    main(args)
