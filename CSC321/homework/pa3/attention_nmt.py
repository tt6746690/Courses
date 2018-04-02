import os
import pdb
import argparse
import pickle as pkl

from collections import defaultdict

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Local imports
import utils
import models


TEST_WORD_ATTN = 'street'
TEST_SENTENCE = 'the air conditioning is working'
# ethay airway onditioningcay isway orkingway


def read_lines(filename):
    """Read a file and split it into lines.
    """
    lines = open(filename).read().strip().lower().split('\n')
    return lines


def read_pairs(filename):
    """Reads lines that consist of two words, separated by a space.

    Returns:
        source_words: A list of the first word in each line of the file.
        target_words: A list of the second word in each line of the file.
    """
    lines = read_lines(filename)
    source_words, target_words = [], []
    for line in lines:
        line = line.strip()
        if line:
            source, target = line.split()
            source_words.append(source)
            target_words.append(target)
    return source_words, target_words


def all_alpha_or_dash(s):
    """Helper function to check whether a string is alphabetic, allowing dashes '-'.
    """
    return all(c.isalpha() or c == '-' for c in s)


def filter_lines(lines):
    """Filters lines to consist of only alphabetic characters or dashes "-".
    """
    return [line for line in lines if all_alpha_or_dash(line)]


def load_data():
    """Loads (English, Pig-Latin) word pairs, and creates mappings from characters to indexes.
    """

    source_lines, target_lines = read_pairs('pig_latin_data.txt')

    # Filter lines
    source_lines = filter_lines(source_lines)
    target_lines = filter_lines(target_lines)

    all_characters = set(''.join(source_lines)) | set(''.join(target_lines))

    # Create a dictionary mapping each character to a unique index
    char_to_index = { char: index for (index, char) in enumerate(sorted(list(all_characters))) }

    # Add start and end tokens to the dictionary
    start_token = len(char_to_index)
    end_token = len(char_to_index) + 1
    char_to_index['SOS'] = start_token
    char_to_index['EOS'] = end_token

    # Create the inverse mapping, from indexes to characters (used to decode the model's predictions)
    index_to_char = { index: char for (char, index) in char_to_index.items() }

    # Store the final size of the vocabulary
    vocab_size = len(char_to_index)

    line_pairs = list(set(zip(source_lines, target_lines)))  # Python 3

    idx_dict = { 'char_to_index': char_to_index,
                 'index_to_char': index_to_char,
                 'start_token': start_token,
                 'end_token': end_token }

    return line_pairs, vocab_size, idx_dict


def create_dict(pairs):
    """Creates a mapping { (source_length, target_length): [list of (source, target) pairs]
    This is used to make batches: each batch consists of two parallel tensors, one containing
    all source indexes and the other containing all corresponding target indexes.
    Within a batch, all the source words are the same length, and all the target words are
    the same length.
    """
    unique_pairs = list(set(pairs))  # Find all unique (source, target) pairs

    d = defaultdict(list)
    for (s,t) in unique_pairs:
        d[(len(s), len(t))].append((s,t))

    return d


def save_loss_plot(train_losses, val_losses, opts):
    """Saves a plot of the training and validation loss curves.
    """
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses)
    plt.plot(range(len(val_losses)), val_losses)
    plt.title('BS={}, nhid={}'.format(opts.batch_size, opts.hidden_size), fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(opts.checkpoint_path, 'loss_plot.pdf'))
    plt.close()


def checkpoint(encoder, decoder, idx_dict, opts):
    """Saves the current encoder and decoder models, along with idx_dict, which
    contains the char_to_index and index_to_char mappings, and the start_token
    and end_token values.
    """
    with open(os.path.join(opts.checkpoint_path, 'encoder.pt'), 'wb') as f:
        torch.save(encoder, f)

    with open(os.path.join(opts.checkpoint_path, 'decoder.pt'), 'wb') as f:
        torch.save(decoder, f)

    with open(os.path.join(opts.checkpoint_path, 'idx_dict.pkl'), 'wb') as f:
        pkl.dump(idx_dict, f)


def evaluate(data_dict, encoder, decoder, idx_dict, criterion, opts):
    """Evaluates the model on a held-out validation or test set.

    Arguments:
        data_dict: The validation/test word pairs, organized by source and target lengths.
        encoder: An encoder model to produce annotations for each step of the input sequence.
        decoder: A decoder model (with or without attention) to generate output tokens.
        idx_dict: Contains char-to-index and index-to-char mappings, and start & end token indexes.
        criterion: Used to compute the CrossEntropyLoss for each decoder output.
        opts: The command-line arguments.

    Returns:
        mean_loss: The average loss over all batches from data_dict.
    """

    start_token = idx_dict['start_token']
    end_token = idx_dict['end_token']
    char_to_index = idx_dict['char_to_index']

    losses = []

    for key in data_dict:

        input_strings, target_strings = zip(*data_dict[key])
        input_tensors = [torch.LongTensor(utils.string_to_index_list(s, char_to_index, end_token)) for s in input_strings]
        target_tensors = [torch.LongTensor(utils.string_to_index_list(s, char_to_index, end_token)) for s in target_strings]

        num_tensors = len(input_tensors)
        num_batches = int(np.ceil(num_tensors / float(opts.batch_size)))

        for i in range(num_batches):

            start = i * opts.batch_size
            end = start + opts.batch_size

            inputs = utils.to_var(torch.stack(input_tensors[start:end]), opts.cuda)
            targets = utils.to_var(torch.stack(target_tensors[start:end]), opts.cuda)

            # The batch size may be different in each epoch
            BS = inputs.size(0)

            encoder_annotations, encoder_hidden = encoder(inputs)

            # The final hidden state of the encoder becomes the initial hidden state of the decoder
            decoder_hidden = encoder_hidden
            start_vector = torch.ones(BS).long().unsqueeze(1) * start_token  # BS x 1
            decoder_input = utils.to_var(start_vector, opts.cuda)  # BS x 1

            loss = 0.0

            seq_len = targets.size(1)  # Gets seq_len from BS x seq_len

            for i in range(seq_len):
                decoder_output, decoder_hidden, attention_weights = decoder(decoder_input, decoder_hidden, encoder_annotations)

                current_target = targets[:,i]
                loss += criterion(decoder_output, current_target)  # cross entropy between the decoder distribution and GT

                ni = F.softmax(decoder_output, dim=1).data.max(1)[1]

                decoder_input = targets[:,i].unsqueeze(1)

            loss /= float(seq_len)
            losses.append(loss.data[0])

    mean_loss = np.mean(losses)

    return mean_loss


def training_loop(train_dict, val_dict, idx_dict, encoder, decoder, criterion, optimizer, opts):
    """Runs the main training loop; evaluates the model on the val set every epoch.
        * Prints training and val loss each epoch.
        * Prints qualitative translation results each epoch using TEST_SENTENCE
        * Saves an attention map for TEST_WORD_ATTN each epoch

    Arguments:
        train_dict: The training word pairs, organized by source and target lengths.
        val_dict: The validation word pairs, organized by source and target lengths.
        idx_dict: Contains char-to-index and index-to-char mappings, and start & end token indexes.
        encoder: An encoder model to produce annotations for each step of the input sequence.
        decoder: A decoder model (with or without attention) to generate output tokens.
        criterion: Used to compute the CrossEntropyLoss for each decoder output.
        optimizer: Implements a step rule to update the parameters of the encoder and decoder.
        opts: The command-line arguments.
    """

    start_token = idx_dict['start_token']
    end_token = idx_dict['end_token']
    char_to_index = idx_dict['char_to_index']

    loss_log = open(os.path.join(opts.checkpoint_path, 'loss_log.txt'), 'w')

    best_val_loss = 1e6
    train_losses = []
    val_losses = []

    for epoch in range(opts.nepochs):

        optimizer.param_groups[0]['lr'] *= opts.lr_decay

        epoch_losses = []

        for key in train_dict:

            input_strings, target_strings = zip(*train_dict[key])
            input_tensors = [torch.LongTensor(utils.string_to_index_list(s, char_to_index, end_token)) for s in input_strings]
            target_tensors = [torch.LongTensor(utils.string_to_index_list(s, char_to_index, end_token)) for s in target_strings]

            num_tensors = len(input_tensors)
            num_batches = int(np.ceil(num_tensors / float(opts.batch_size)))

            for i in range(num_batches):

                start = i * opts.batch_size
                end = start + opts.batch_size

                inputs = utils.to_var(torch.stack(input_tensors[start:end]), opts.cuda)
                targets = utils.to_var(torch.stack(target_tensors[start:end]), opts.cuda)

                # The batch size may be different in each epoch
                BS = inputs.size(0)

                encoder_annotations, encoder_hidden = encoder(inputs)

                # The last hidden state of the encoder becomes the first hidden state of the decoder
                decoder_hidden = encoder_hidden

                start_vector = torch.ones(BS).long().unsqueeze(1) * start_token  # BS x 1 --> 16x1  CHECKED
                decoder_input = utils.to_var(start_vector, opts.cuda)  # BS x 1 --> 16x1  CHECKED

                loss = 0.0

                seq_len = targets.size(1)  # Gets seq_len from BS x seq_len

                use_teacher_forcing = np.random.rand() < opts.teacher_forcing_ratio

                for i in range(seq_len):
                    decoder_output, decoder_hidden, attention_weights = decoder(decoder_input, decoder_hidden, encoder_annotations)

                    current_target = targets[:,i]
                    loss += criterion(decoder_output, current_target)  # cross entropy between the decoder distribution and GT
                    ni = F.softmax(decoder_output, dim=1).data.max(1)[1]

                    if use_teacher_forcing:
                        # With teacher forcing, use the ground-truth token to condition the next step
                        decoder_input = targets[:,i].unsqueeze(1)
                    else:
                        # Without teacher forcing, use the model's own predictions to condition the next step
                        decoder_input = utils.to_var(ni.unsqueeze(1), opts.cuda)

                loss /= float(seq_len)
                epoch_losses.append(loss.data[0])

                # Zero gradients
                optimizer.zero_grad()

                # Compute gradients
                loss.backward()

                # Update the parameters of the encoder and decoder
                optimizer.step()

        train_loss = np.mean(epoch_losses)
        val_loss = evaluate(val_dict, encoder, decoder, idx_dict, criterion, opts)

        if val_loss < best_val_loss:
            checkpoint(encoder, decoder, idx_dict, opts)

        if not opts.no_attention:
            # Save attention maps for the fixed word TEST_WORD_ATTN throughout training
            utils.visualize_attention(TEST_WORD_ATTN,
                                      encoder,
                                      decoder,
                                      idx_dict,
                                      opts,
                                      save=os.path.join(opts.checkpoint_path, 'train_attns/attn-epoch-{}.png'.format(epoch)))

        gen_string = utils.translate_sentence(TEST_SENTENCE, encoder, decoder, idx_dict, opts)
        print("Epoch: {:3d} | Train loss: {:.3f} | Val loss: {:.3f} | Gen: {:20s}".format(epoch, train_loss, val_loss, gen_string))

        loss_log.write('{} {} {}\n'.format(epoch, train_loss, val_loss))
        loss_log.flush()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        save_loss_plot(train_losses, val_losses, opts)


def print_data_stats(line_pairs, vocab_size, idx_dict):
    """Prints example word pairs, the number of data points, and the vocabulary.
    """
    print('=' * 80)
    print('Data Stats'.center(80))
    print('-' * 80)
    for pair in line_pairs[:5]:
        print(pair)
    print('Num unique word pairs: {}'.format(len(line_pairs)))
    print('Vocabulary: {}'.format(idx_dict['char_to_index'].keys()))
    print('Vocab size: {}'.format(vocab_size))
    print('=' * 80)


def main(opts):

    line_pairs, vocab_size, idx_dict = load_data()
    print_data_stats(line_pairs, vocab_size, idx_dict)

    # Split the line pairs into an 80% train and 20% val split
    num_lines = len(line_pairs)
    num_train = int(0.8 * num_lines)
    train_pairs, val_pairs = line_pairs[:num_train], line_pairs[num_train:]

    # Group the data by the lengths of the source and target words, to form batches
    train_dict = create_dict(train_pairs)
    val_dict = create_dict(val_pairs)

    ##########################################################################
    ### Setup: Create Encoder, Decoder, Learning Criterion, and Optimizers ###
    ##########################################################################
    encoder = models.GRUEncoder(vocab_size=vocab_size, hidden_size=opts.hidden_size, opts=opts)

    if opts.no_attention:
        decoder = models.NoAttentionDecoder(vocab_size=vocab_size, hidden_size=opts.hidden_size)
    else:
        decoder = models.AttentionDecoder(vocab_size=vocab_size, hidden_size=opts.hidden_size)

    if opts.cuda:
        encoder.cuda()
        decoder.cuda()
        print("Moved models to GPU!")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=opts.learning_rate)

    try:
        training_loop(train_dict, val_dict, idx_dict, encoder, decoder, criterion, optimizer, opts)
    except KeyboardInterrupt:
        print('Exiting early from training.')


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Training hyper-parameters
    parser.add_argument('--nepochs', type=int, default=100,
                        help='The max number of epochs.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='The number of examples in a batch.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='The learning rate (default 0.01)')
    parser.add_argument('--lr_decay', type=float, default=0.99,
                        help='Set the learning rate decay factor.')
    parser.add_argument('--hidden_size', type=int, default=10,
                        help='The size of the GRU hidden state.')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5,
                        help='The proportion of the time teacher forcing is used.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Set the directry to store the best model checkpoints.')
    parser.add_argument('--no_attention', action='store_true', default=False,
                        help='Use the NoAttentionDecoder model.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Choose whether to use GPU.')

    return parser


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    print_opts(opts)

    model_name = 'h{}-bs{}'.format(opts.hidden_size, opts.batch_size)
    opts.checkpoint_path = os.path.join(opts.checkpoint_dir, model_name)

    utils.create_dir_if_not_exists(opts.checkpoint_path)
    utils.create_dir_if_not_exists(os.path.join(opts.checkpoint_path, 'train_attns'))

    main(opts)
