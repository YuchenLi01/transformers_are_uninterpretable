"""
Utilities for determining paths to corpora, results, models
given config dictionaries describing an experiment, as well
as determining canonical vocabulary ordering
"""

import os
import string


def get_identifier_iterator():
    """ Returns an iterator to provide unique ids to bracket types.
    """
    ids = iter(list(string.ascii_lowercase))
    k = 1
    while True:
        try:
            str_id = next(ids)
        except StopIteration:
            ids = iter(list(string.ascii_lowercase))
            k += 1
            str_id = next(ids)
        yield str_id*k


def get_vocab_of_bracket_types(bracket_types):
    """ Returns the vocabulary corresponding to the number of brackets.

    There are bracket_types open brackets, bracket_types close brackets,
    START, and END.
    Arguments:
      bracket_types: int (k in Dyck-(k,m))
    Returns:
      Dictionary mapping symbol string  s to int ids.
    """
    id_iterator = get_identifier_iterator()
    ids = [next(id_iterator) for x in range(bracket_types)]
    vocab = {x: c for c, x in enumerate(
                ['PAD', 'MASK', 'START', 'END']
                + ['(' + id_str for id_str in ids]
                + [id_str + ')' for id_str in ids]
            )}
    closing_bracket_ids = {vocab[id_str + ')'] for id_str in ids}
    return vocab, ids, closing_bracket_ids


def get_results_dir_of_args(args):
    """
    Takes a (likely yaml-defined) argument dictionary
    and returns the directory to which results of the
    experiment defined by the arguments will be saved
    """
    return args['reporting']['reporting_loc']


def get_corpus_paths_of_args(args):
    paths = {
            'train': args['corpus']['train_corpus_loc'],
            'dev': args['corpus']['dev_corpus_loc'],
            'train_out': None,
            'dev_out': None,
            'test_out': None,
        }
    if 'train_output_loc' in args['corpus']:
        paths['train_out'] = args['corpus']['train_output_loc']
    if 'dev_output_loc' in args['corpus']:
        paths['dev_out'] = args['corpus']['dev_output_loc']
    return paths


def get_lm_path_of_args(args):
    results_dir = get_results_dir_of_args(args)
    return os.path.join(results_dir, args['name']+'.params')

def get_arg_path_of_args(args):
    results_dir = get_results_dir_of_args(args)
    return os.path.join(results_dir, args['name']+'.args')

def get_dataset_path_of_args(args):
    results_dir = get_results_dir_of_args(args)
    return os.path.join(results_dir, args['name']+'.dataset')

def get_fail_path_of_args(args):
    results_dir = get_results_dir_of_args(args)
    return os.path.join(results_dir, args['name']+'.failcase')

def get_acc_path_of_args(args, split):
    results_dir = get_results_dir_of_args(args)
    return os.path.join(results_dir, args['name']+f".{split}_acc")