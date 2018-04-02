# CSC 321, Assignment 4
#
# This is a script to check whether the outputs of your CycleGenerator, DCDiscriminator, and
# CycleGenerator models produce the expected outputs.
#
# NOTE THAT THIS MODEL CHECKER IS PROVIDED FOR CONVENIENCE ONLY, AND MAY PRODUCE FALSE NEGATIVES.
# DO NOT USE THIS AS THE ONLY WAY TO CHECK THAT YOUR MODEL IS CORRECT.
#
# Usage:
# ======
#
#    python model_checker.py
#

import warnings
warnings.filterwarnings("ignore")

# Torch imports
import torch
from torch.autograd import Variable

# Numpy
import numpy as np

# Local imports
from models import DCGenerator, DCDiscriminator, CycleGenerator


def count_parameters(model):
    """Finds the total number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sample_noise(dim):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (1, dim, 1, 1) containing uniform
      random noise in the range (-1, 1).
    """
    return Variable(torch.rand(1, dim) * 2 - 1).unsqueeze(2).unsqueeze(3)


def check_dc_generator():
    """Checks the output and number of parameters of the DCGenerator class.
    """
    state = torch.load('checker_files/dc_generator.pt')

    G = DCGenerator(noise_size=100, conv_dim=32)
    G.load_state_dict(state['state_dict'])
    noise = state['input']
    dc_generator_expected = state['output']

    output = G(noise)
    output_np = output.data.cpu().numpy()

    if np.allclose(output_np, dc_generator_expected):
        print('DCGenerator output: EQUAL')
    else:
        print('DCGenerator output: NOT EQUAL')

    num_params = count_parameters(G)
    expected_params = 370624

    print('DCGenerator #params = {}, expected #params = {}, {}'.format(
          num_params, expected_params, 'EQUAL' if num_params == expected_params else 'NOT EQUAL'))

    print('-' * 80)


def check_dc_discriminator():
    """Checks the output and number of parameters of the DCDiscriminator class.
    """
    state = torch.load('checker_files/dc_discriminator.pt')

    D = DCDiscriminator(conv_dim=32)
    D.load_state_dict(state['state_dict'])
    images = state['input']
    dc_discriminator_expected = state['output']

    output = D(images)
    output_np = output.data.cpu().numpy()

    if np.allclose(output_np, dc_discriminator_expected):
        print('DCDiscriminator output: EQUAL')
    else:
        print('DCDiscriminator output: NOT EQUAL')

    num_params = count_parameters(D)
    expected_params = 167872

    print('DCDiscriminator #params = {}, expected #params = {}, {}'.format(
          num_params, expected_params, 'EQUAL' if num_params == expected_params else 'NOT EQUAL'))

    print('-' * 80)


def check_cycle_generator():
    """Checks the output and number of parameters of the CycleGenerator class.
    """
    state = torch.load('checker_files/cycle_generator.pt')

    G_XtoY = CycleGenerator(conv_dim=32, init_zero_weights=False)
    G_XtoY.load_state_dict(state['state_dict'])
    images = state['input']
    cycle_generator_expected = state['output']

    output = G_XtoY(images)
    output_np = output.data.cpu().numpy()

    if np.allclose(output_np, cycle_generator_expected):
        print('CycleGenerator output: EQUAL')
    else:
        print('CycleGenerator output: NOT EQUAL')

    num_params = count_parameters(G_XtoY)
    expected_params = 105856

    print('CycleGenerator #params = {}, expected #params = {}, {}'.format(
          num_params, expected_params, 'EQUAL' if num_params == expected_params else 'NOT EQUAL'))

    print('-' * 80)


if __name__ == '__main__':

    try:
        check_dc_generator()
    except:
        print('Crashed while checking DCGenerator. Maybe not implemented yet?')

    try:
        check_dc_discriminator()
    except:
        print('Crashed while checking DCDiscriminator. Maybe not implemented yet?')

    try:
        check_cycle_generator()
    except:
        print('Crashed while checking CycleGenerator. Maybe not implemented yet?')
