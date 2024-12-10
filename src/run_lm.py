""" Trains/runs a language model on data available tokenized sentence-per-line format.

The main interface to running experiments with this codebase.

Usage:
      python rnns_stacks/run_lm.py <config.yaml>
"""

import torch
import yaml
import os
from tqdm import tqdm
from argparse import ArgumentParser
from dataset import Dataset
from classification_dataset import ClassifyDataset
from training_regimen import train
import utils
import transformer
import wandb


def create_args(config_file):
    args = yaml.load(open(config_file))
    args['training']['learning_rate'] = float(args['training']['learning_rate'])
    args['training']['weight_decay'] = float(args['training']['weight_decay'])
    args['training']['dropout'] = float(args['training']['dropout'])

    if args['lm']['lm_type'] in {'BertForMaskedLM', 'BertForMaskedLMCustom'}:
        args['name'] = f"_mask_{args['training']['mask_prob']}"
    elif args['lm']['lm_type'] in {'GPT2LMHeadModel','GPT2LMHeadModelCustom'}:
        args['name'] = ""
    else:
        raise NotImplementedError('Model not supported.')

    if args['language']['name'] == 'dyck':
        if(args['training']['weight_decay'] > 0):
            args['name'] = "dyck_k{}_m{}_{}_lr{}_wd_{}_hiddenlayers{}_heads{}_hiddendim{}_{}_{}_{}_dropout{}".format(
                args['language']['bracket_types'],
                args['language']['train_max_stack_depth'],
                args['lm']['lm_type'],
                args['training']['learning_rate'],
                args['training']['weight_decay'],
                args['lm']['num_layers'],
                args['lm']['num_heads'],
                args['lm']['hidden_dim'],
                args['lm']['embedding_type'],
                args['lm']['token_embedding_type'],
                args['training']['objective'],
                args['training']['dropout'],
            ) + args['name']
        else:
            args['name'] = "dyck_k{}_m{}_{}_lr{}_hiddenlayers{}_heads{}_hiddendim{}_{}_{}_{}_dropout{}".format(
                args['language']['bracket_types'],
                args['language']['train_max_stack_depth'],
                args['lm']['lm_type'],
                args['training']['learning_rate'],
                args['lm']['num_layers'],
                args['lm']['num_heads'],
                args['lm']['hidden_dim'],
                args['lm']['embedding_type'],
                args['lm']['token_embedding_type'],
                args['training']['objective'],
                args['training']['dropout'],
            ) + args['name']
    else:
        raise NotImplementedError('Language not supported.')

    if args['training']['mask_correct_prob'] > 0:
        args['name'] += f"_correct{args['training']['mask_correct_prob']}"
    if args['training']['mask_random_prob'] > 0:
        args['name'] += f"_random{args['training']['mask_random_prob']}"
        
    # Customize Part
    # This is added to make sure the config is back compatible
    if(args['lm']['lm_type'] == 'BertForMaskedLMCustom'):
        if (not args['lm']['residual']) and (not args['lm']['attn_output_fc']) and (not args['lm']['bert_intermediate']) \
                and (not args['lm']['bert_output']) and (not args['lm']['bert_head_transform'])\
                and (not args['lm']['layer_norm']):
            args['name'] += '_noMany'
        else:
            if not args['lm']['residual']:
                args['name'] += '_noRes'
            if not args['lm']['attn_output_fc']:
                args['name'] += '_noAttnOutFC'
            if not args['lm']['bert_intermediate']:
                args['name'] += '_noBertIntermediate'
            if not args['lm']['bert_output']:
                args['name'] += '_noBertOutput'
            if not args['lm']['bert_head_transform']:
                args['name'] += '_noBertHeadTransform'
            if not args['lm']['layer_norm']:
                args['name'] += '_noLayerNorm'
        if args['lm']['freeze_uniform_attention']:
            args['name'] += '_freezeUniformAttention'
        if(args['lm']['freeze_decoder_to_I']):
            args['name'] += '_identityDecoder'
    elif (args['lm']['lm_type'] == 'GPT2LMHeadModelCustom'):
        if (not args['lm']['residual']) and (not args['lm']['attn_output_fc']) and (not args['lm']['gpt_intermediate']) \
                and (not args['lm']['layer_norm']):
            args['name'] += '_noMany'
        else:
            if not args['lm']['residual']:
                args['name'] += '_noRes'
            if not args['lm']['attn_output_fc']:
                args['name'] += '_noAttnOutFC'
            if not args['lm']['gpt_intermediate']:
                args['name'] += '_noFFN'
            if not args['lm']['layer_norm']:
                args['name'] += '_noLayerNorm'
        if(args['lm']['freeze_decoder_to_I']):
            args['name'] += '_identityDecoder'    
        if args['lm']['freeze_uniform_attention']:
            args['name'] += '_freezeUniformAttention'
        if args['lm']['one_layer_norm']:
            args['name'] += '_oneLayerNorm'
        if args['lm']['first_residual']:
            args['name'] += '_firstResidue'
        if args['lm']['another_first_residual']:
            args['name'] += '_anotherFirstResidue'   
    else:
        assert args['lm']['residual'] and args['lm']['attn_output_fc'] and args['lm']['layer_norm'] \
               and not args['lm']['freeze_uniform_attention'] \
               and not args['lm']['freeze_decoder_to_I'], \
               'Altering non-custom models will have no effect'
    if(args['training']['optimizer'] == 'SAM'):
        args['name'] += f"_SAM{args['training']['rho']}"   
    if(args['training']['seed'] > 0):
        args['name'] += '_{}'.format(args['training']['seed'])


    # Determine whether CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args['device'] = device
    return args


def load_dataset(args):
    if(args['training']['objective'] not in {'classify'}):
        dataset = Dataset(args)
    else:
        dataset = ClassifyDataset(args)
    return dataset


def init_lm(args):
    return transformer.PytorchTransformerModel(args)


if __name__ == '__main__':
    argp = ArgumentParser()
    argp.add_argument('config')
    args = argp.parse_args()
    args = create_args(args.config)

    # Must load dataset first, since args['language']['vocab_size'] depends on data
    dataset = load_dataset(args)    
    name_base = args['name']
    for experiment_index in range(args['experiment']['repeat']):
        args['name'] = name_base + str(experiment_index)
        if args['name'] in {
            # list of runs to skip e.g. because they have been completed previously
        }:
            print(f"Skipped {args['name']}")
            continue

        # Construct the language model
        print('Construct the language model with args', args)
        lm_model = init_lm(args)

        # Prepare to write results
        output_dir = utils.get_results_dir_of_args(args)
        tqdm.write('Writing results to {}'.format(output_dir))
        os.makedirs(utils.get_results_dir_of_args(args), exist_ok=True)

        wandb.init(
            project="dyck_transformer_dynamics",
            name=args['name'],
            reinit=True,
        )
        wandb.config = args

        # Train and load most recent parameters
        train(args, lm_model, dataset.get_train_dataloader(), dataset.get_dev_dataloader(),
              steps_between_logging=5,
              steps_between_evals=50,  # original 1000. more frequent eval to track learning.
              dataset=dataset,
              )

    # lm_model.load_state_dict(torch.load(utils.get_lm_path_of_args(args)))

    # # Evaluate language model (only implemented for Dyck)
    # if args['language']['name'] == 'dyck':
    #     reporter.run_evals(args, lm_model, dataset, 'test20')
    #     reporter.run_evals(args, lm_model, dataset, 'dev')
    #     reporter.run_evals(args, lm_model, dataset, 'test')
