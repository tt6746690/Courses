"""Script to visualize attention maps using a pre-trained model.

    Usage: python translate_no_attn.py --load=checkpoints/no-attn/py2
"""

import os
import pdb
import sys
import argparse
import pickle as pkl

import warnings
warnings.filterwarnings("ignore")

import numpy as np

import torch

# Local imports
import utils


words = [
        # 'roomba',
        #  'concert',
        #  'hello',
        #  'table'
         # Add your own words here!

         # short words + starts with consonant
         'seat',
         'fact',
         'how',

         # short words + starts with vowel
         'able',
         'eight',
         'aches',

         # short words + starts with sh  
         'shops',
         'shew',
         'shades',

         # long words + starts with consonant
         'misconstruction',
         'prepossessing',
         'reasonableness',

        # long words + starts with vowel
         'instantaneously',
         'insurmountable',
         'acknowledgments',
        ]


def load(opts):
    encoder = torch.load(os.path.join(opts.load, 'encoder.pt'))
    decoder = torch.load(os.path.join(opts.load, 'decoder.pt'))
    idx_dict = pkl.load(open(os.path.join(opts.load, 'idx_dict.pkl'), 'rb'))
    return encoder, decoder, idx_dict


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default='checkpoints/no-attn/py2', help='Path to checkpoint directory.')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use GPU.')
    return parser


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    if sys.version_info[0] == 3:
        opts.load = 'checkpoints/no-attn/py3'

    encoder, decoder, idx_dict = load(opts)

    for word in words:
        translated = utils.translate(word,
                                     encoder,
                                     decoder,
                                     idx_dict,
                                     opts)

        print('{} --> {}'.format(word, translated))
