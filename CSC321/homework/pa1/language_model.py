import collections
import cPickle
import numpy as np
nax = np.newaxis
import pylab

import tsne

TINY = 1e-30


DEFAULT_TRAINING_CONFIG = {'batch_size': 100,                    # the size of a mini-batch
                           'learning_rate': 0.1,                 # the learning rate
                           'momentum': 0.9,                      # the decay parameter for the momentum vector
                           'epochs': 50,                         # the maximum number of epochs to run
                           'init_wt': 0.01,                      # the standard deviation of the initial random weights
                           'context_len': 3,                     # the number of context words used
                           'show_training_CE_after': 100,        # measure training error after this many mini-batches
                           'show_validation_CE_after': 1000,     # measure validation error after this many mini-batches
                           }


def get_batches(inputs, targets, batch_size, shuffle=True):
    """Divide a dataset (usually the training set) into mini-batches of a given size. This is a
    'generator', i.e. something you can use in a for loop. You don't need to understand how it
    works to do the assignment."""
    
    if inputs.shape[0] % batch_size != 0:
        raise RuntimeError('The number of data points must be a multiple of the batch size.')
    num_batches = inputs.shape[0] // batch_size

    if shuffle:
        idxs = np.random.permutation(inputs.shape[0])
        inputs = inputs[idxs, :]
        targets = targets[idxs]

    for m in range(num_batches):
        yield inputs[m*batch_size:(m+1)*batch_size, :], \
              targets[m*batch_size:(m+1)*batch_size]
    



class Params(object):
    """A class representing the trainable parameters of the model. This class has five fields:
    
           word_embedding_weights, a matrix of size N_V x D, where N_V is the number of words in the vocabulary
                   and D is the embedding dimension.
           embed_to_hid_weights, a matrix of size N_H x 3D, where N_H is the number of hidden units. The first D
                   columns represent connections from the embedding of the first context word, the next D columns
                   for the second context word, and so on.
           hid_bias, a vector of length N_H
           hid_to_output_weights, a matrix of size N_V x N_H
           output_bias, a vector of length N_V"""
    
    def __init__(self, word_embedding_weights, embed_to_hid_weights, hid_to_output_weights,
                 hid_bias, output_bias):
        self.word_embedding_weights = word_embedding_weights
        self.embed_to_hid_weights = embed_to_hid_weights
        self.hid_to_output_weights = hid_to_output_weights
        self.hid_bias = hid_bias
        self.output_bias = output_bias

    def copy(self):
        return self.__class__(self.word_embedding_weights.copy(), self.embed_to_hid_weights.copy(),
                              self.hid_to_output_weights.copy(), self.hid_bias.copy(), self.output_bias.copy())

    @classmethod
    def zeros(cls, vocab_size, context_len, embedding_dim, num_hid):
        """A constructor which initializes all weights and biases to 0."""
        word_embedding_weights = np.zeros((vocab_size, embedding_dim))
        embed_to_hid_weights = np.zeros((num_hid, context_len * embedding_dim))
        hid_to_output_weights = np.zeros((vocab_size, num_hid))
        hid_bias = np.zeros(num_hid)
        output_bias = np.zeros(vocab_size)
        return cls(word_embedding_weights, embed_to_hid_weights, hid_to_output_weights,
                   hid_bias, output_bias)

    @classmethod
    def random_init(cls, init_wt, vocab_size, context_len, embedding_dim, num_hid):
        """A constructor which initializes weights to small random values and biases to 0."""
        word_embedding_weights = np.random.normal(0., init_wt, size=(vocab_size, embedding_dim))
        embed_to_hid_weights = np.random.normal(0., init_wt, size=(num_hid, context_len * embedding_dim))
        hid_to_output_weights = np.random.normal(0., init_wt, size=(vocab_size, num_hid))
        hid_bias = np.zeros(num_hid)
        output_bias = np.zeros(vocab_size)
        return cls(word_embedding_weights, embed_to_hid_weights, hid_to_output_weights,
                   hid_bias, output_bias)


    ###### The functions below are Python's somewhat oddball way of overloading operators, so that
    ###### we can do arithmetic on Params instances. You don't need to understand this to do the assignment.

    def __mul__(self, a):
        return self.__class__(a * self.word_embedding_weights,
                              a * self.embed_to_hid_weights,
                              a * self.hid_to_output_weights,
                              a * self.hid_bias,
                              a * self.output_bias)

    def __rmul__(self, a):
        return self * a

    def __add__(self, other):
        return self.__class__(self.word_embedding_weights + other.word_embedding_weights,
                              self.embed_to_hid_weights + other.embed_to_hid_weights,
                              self.hid_to_output_weights + other.hid_to_output_weights,
                              self.hid_bias + other.hid_bias,
                              self.output_bias + other.output_bias)

    def __sub__(self, other):
        return self + -1. * other
                              

class Activations(object):
    """A class representing the activations of the units in the network. This class has three fields:

        embedding_layer, a matrix of B x 3D matrix (where B is the batch size and D is the embedding dimension),
                representing the activations for the embedding layer on all the cases in a batch. The first D
                columns represent the embeddings for the first context word, and so on.
        hidden_layer, a B x N_H matrix representing the hidden layer activations for a batch
        output_layer, a B x N_V matrix representing the output layer activations for a batch"""
    
    def __init__(self, embedding_layer, hidden_layer, output_layer):
        self.embedding_layer = embedding_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer


class Model(object):
    """A class representing the language model itself. This class contains various methods used in training
    the model and visualizing the learned representations. It has two fields:

        params, a Params instance which contains the model parameters
        vocab, a list containing all the words in the dictionary; vocab[0] is the word with index
               0, and so on."""
    
    def __init__(self, params, vocab):
        self.params = params
        self.vocab = vocab

        self.vocab_size = len(vocab)
        self.embedding_dim = self.params.word_embedding_weights.shape[1]
        self.embedding_layer_dim = self.params.embed_to_hid_weights.shape[1]
        self.context_len = self.embedding_layer_dim // self.embedding_dim
        self.num_hid = self.params.embed_to_hid_weights.shape[0]

    def copy(self):
        return self.__class__(self.params.copy(), self.vocab[:])

    @classmethod
    def random_init(cls, init_wt, vocab, context_len, embedding_dim, num_hid):
        """Constructor which randomly initializes the weights to Gaussians with standard deviation init_wt
        and initializes the biases to all zeros."""
        params = Params.random_init(init_wt, len(vocab), context_len, embedding_dim, num_hid)
        return Model(params, vocab)

    def indicator_matrix(self, targets):
        """Construct a matrix where the kth entry of row i is 1 if the target for example
        i is k, and all other entries are 0."""
        batch_size = targets.size
        expanded_targets = np.zeros((batch_size, len(self.vocab)))
        expanded_targets[np.arange(batch_size), targets] = 1.
        return expanded_targets

    def compute_loss_derivative(self, output_activations, expanded_target_batch):
        """Compute the derivative of the cross-entropy loss function with respect to the inputs
        to the output units. In particular, the output layer computes the softmax

            y_i = e^{z_i} / \sum_j e^{z_j}.

        This function should return a batch_size x vocab_size matrix, where the (i, j) entry
        is dC / dz_j computed for the ith training case, where C is the loss function

            C = -sum(t_i log y_i).

        The arguments are as follows:

            output_activations - the activations of the output layer, i.e. the y_i's.
            expanded_target_batch - each row is the indicator vector for a target word,
                i.e. the (i, j) entry is 1 if the i'th word is j, and 0 otherwise."""
        
        ###########################   YOUR CODE HERE  ##############################


        ############################################################################


    def compute_loss(self, output_activations, expanded_target_batch):
        """Compute the total loss over a mini-batch. expanded_target_batch is the matrix obtained
        by calling indicator_matrix on the targets for the batch."""
        return -np.sum(expanded_target_batch * np.log(output_activations + TINY))

    def compute_activations(self, inputs):
        """Compute the activations on a batch given the inputs. Returns an Activations instance.
        You should try to read and understand this function, since this will give you clues for
        how to implement back_propagate."""
        
        batch_size = inputs.shape[0]
        if inputs.shape[1] != self.context_len:
            raise RuntimeError('Dimension of the input vectors should be {}, but is instead {}'.format(
                self.context_len, inputs.shape[1]))

        # Embedding layer
        # Look up the input word indies in the word_embedding_weights matrix
        embedding_layer_state = np.zeros((batch_size, self.embedding_layer_dim))
        for i in range(self.context_len):
            embedding_layer_state[:, i*self.embedding_dim:(i+1)*self.embedding_dim] = \
                                     self.params.word_embedding_weights[inputs[:, i], :]

        # Hidden layer
        inputs_to_hid = np.dot(embedding_layer_state, self.params.embed_to_hid_weights.T) + \
                        self.params.hid_bias
        # Apply logistic activation function
        hidden_layer_state = 1. / (1. + np.exp(-inputs_to_hid))

        # Output layer
        inputs_to_softmax = np.dot(hidden_layer_state, self.params.hid_to_output_weights.T) + \
                            self.params.output_bias

        # Subtract maximum.
        # Remember that adding or subtracting the same constant from each input to a
        # softmax unit does not affect the outputs. So subtract the maximum to
        # make all inputs <= 0. This prevents overflows when computing their exponents.
        inputs_to_softmax -= inputs_to_softmax.max(1).reshape((-1, 1))

        output_layer_state = np.exp(inputs_to_softmax)
        output_layer_state /= output_layer_state.sum(1).reshape((-1, 1))

        return Activations(embedding_layer_state, hidden_layer_state, output_layer_state)

        

    def back_propagate(self, input_batch, activations, loss_derivative):
        """Compute the gradient of the loss function with respect to the trainable parameters
        of the model. The arguments are as follows:

             input_batch - the indices of the context words
             activations - an Activations class representing the output of Model.compute_activations
             loss_derivative - the matrix of derivatives computed by compute_loss_derivative
             
        Part of this function is already completed, but you need to fill in the derivative
        computations for hid_to_output_weights_grad, output_bias_grad, embed_to_hid_weights_grad,
        and hid_bias_grad. See the documentation for the Params class for a description of what
        these matrices represent."""
        
        # The matrix with values dC / dz_j, where dz_j is the input to the jth hidden unit,
        # i.e. y_j = 1 / (1 + e^{-z_j})
        hid_deriv = np.dot(loss_derivative, self.params.hid_to_output_weights) \
                    * activations.hidden_layer * (1. - activations.hidden_layer)


        ###########################   YOUR CODE HERE  ##############################


        ############################################################################


        # The matrix of derivatives for the embedding layer
        embed_deriv = np.dot(hid_deriv, self.params.embed_to_hid_weights)

        # Embedding layer
        word_embedding_weights_grad = np.zeros((self.vocab_size, self.embedding_dim))
        for w in range(self.context_len):
            word_embedding_weights_grad += np.dot(self.indicator_matrix(input_batch[:, w]).T,
                                                  embed_deriv[:, w*self.embedding_dim:(w+1)*self.embedding_dim])
            
        return Params(word_embedding_weights_grad, embed_to_hid_weights_grad, hid_to_output_weights_grad,
                      hid_bias_grad, output_bias_grad)

    def evaluate(self, inputs, targets, batch_size=100):
        """Compute the average cross-entropy over a dataset.

            inputs: matrix of shape D x N
            targets: one-dimensional matrix of length N"""
        
        ndata = inputs.shape[0]

        total = 0.
        for input_batch, target_batch in get_batches(inputs, targets, batch_size):
            activations = self.compute_activations(input_batch)
            expanded_target_batch = self.indicator_matrix(target_batch)
            cross_entropy = -np.sum(expanded_target_batch * np.log(activations.output_layer + TINY))
            total += cross_entropy

        return total / float(ndata)

    def display_nearest_words(self, word, k=10):
        """List the k words nearest to a given word, along with their distances."""
        
        if word not in self.vocab:
            print 'Word "{}" not in vocabulary.'.format(word)
            return

        # Compute distance to every other word.
        idx = self.vocab.index(word)
        word_rep = self.params.word_embedding_weights[idx, :]
        diff = self.params.word_embedding_weights - word_rep.reshape((1, -1))
        distance = np.sqrt(np.sum(diff**2, axis=1))

        # Sort by distance.
        order = np.argsort(distance)
        order = order[1:1+k]      # The nearest word is the query word itself, skip that.
        for i in order:
            print '{}: {}'.format(self.vocab[i], distance[i])

    def predict_next_word(self, word1, word2, word3, k=10):
        """List the top k predictions for the next word along with their probabilities.
        Inputs:
            word1: The first word as a string.
            word2: The second word as a string.
            word3: The third word as a string.
            k: The k most probable predictions are shown.
        Example usage:
            model.predict_next_word('john', 'might', 'be', 3)
            model.predict_next_word('life', 'in', 'new', 3)"""
            
        if word1 not in self.vocab:
            raise RuntimeError('Word "{}" not in vocabulary.'.format(word1))
        if word2 not in self.vocab:
            raise RuntimeError('Word "{}" not in vocabulary.'.format(word2))
        if word3 not in self.vocab:
            raise RuntimeError('Word "{}" not in vocabulary.'.format(word3))

        idx1, idx2, idx3 = self.vocab.index(word1), self.vocab.index(word2), self.vocab.index(word3)
        input = np.array([idx1, idx2, idx3]).reshape((1, -1))
        activations = self.compute_activations(input)
        prob = activations.output_layer.ravel()
        idxs = np.argsort(prob)[::-1]   # sort descending
        for i in idxs[:k]:
            print '{} {} {} {} Prob: {:1.5f}'.format(word1, word2, word3, self.vocab[i], prob[i])

    def word_distance(self, word1, word2):
        """Compute the distance between the vector representations of two words."""
        
        if word1 not in self.vocab:
            raise RuntimeError('Word "{}" not in vocabulary.'.format(word1))
        if word2 not in self.vocab:
            raise RuntimeError('Word "{}" not in vocabulary.'.format(word2))
        
        idx1, idx2 = self.vocab.index(word1), self.vocab.index(word2)
        word_rep1 = self.params.word_embedding_weights[idx1, :]
        word_rep2 = self.params.word_embedding_weights[idx2, :]
        diff = word_rep1 - word_rep2
        return np.sqrt(np.sum(diff ** 2))

    def tsne_plot(self):
        """Plot a 2-D visualization of the learned representations using t-SNE."""
        
        mapped_X = tsne.tsne(self.params.word_embedding_weights)
        pylab.figure()
        for i, w in enumerate(self.vocab):
            pylab.text(mapped_X[i, 0], mapped_X[i, 1], w)
        pylab.xlim(mapped_X[:, 0].min(), mapped_X[:, 0].max())
        pylab.ylim(mapped_X[:, 1].min(), mapped_X[:, 1].max())
        pylab.show()



_train_inputs = None
_train_targets = None
_vocab = None

def find_occurrences(word1, word2, word3):
    """Lists all the words that followed a given tri-gram in the training set and the number of
    times each one followed it."""

    # cache the data so we don't keep reloading
    global _train_inputs, _train_targets, _vocab
    if _train_inputs is None:
        data_obj = cPickle.load(open('data.pk', 'rb'))
        _vocab = data_obj['vocab']
        _train_inputs, _train_targets = data_obj['train_inputs'], data_obj['train_targets']
    
    if word1 not in _vocab:
        raise RuntimeError('Word "{}" not in vocabulary.'.format(word1))
    if word2 not in _vocab:
        raise RuntimeError('Word "{}" not in vocabulary.'.format(word2))
    if word3 not in _vocab:
        raise RuntimeError('Word "{}" not in vocabulary.'.format(word3))
    
    idx1, idx2, idx3 = _vocab.index(word1), _vocab.index(word2), _vocab.index(word3)
    idxs = np.array([idx1, idx2, idx3])

    matches = np.all(_train_inputs == idxs.reshape((1, -1)), 1)

    if np.any(matches):
        counts = collections.defaultdict(int)
        for m in np.where(matches)[0]:
            counts[_vocab[_train_targets[m]]] += 1

        word_counts = sorted(counts.items(), key=lambda t: t[1], reverse=True)
        print 'The tri-gram "{} {} {}" was followed by the following words in the training set:'.format(
            word1, word2, word3)
        for word, count in word_counts:
            if count > 1:
                print '    {} ({} times)'.format(word, count)
            else:
                print '    {} (1 time)'.format(word)
    else:
        print 'The tri-gram "{} {} {}" did not occur in the training set.'.format(word1, word2, word3)



def train(embedding_dim, num_hid, config=DEFAULT_TRAINING_CONFIG):
    """This is the main training routine for the language model. It takes two parameters:

        embedding_dim, the dimension of the embedding space
        num_hid, the number of hidden units."""
    
    # Load the data
    data_obj = cPickle.load(open('data.pk', 'rb'))
    vocab = data_obj['vocab']
    train_inputs, train_targets = data_obj['train_inputs'], data_obj['train_targets']
    valid_inputs, valid_targets = data_obj['valid_inputs'], data_obj['valid_targets']
    test_inputs, test_targets = data_obj['test_inputs'], data_obj['test_targets']
    
    # Randomly initialize the trainable parameters
    model = Model.random_init(config['init_wt'], vocab, config['context_len'], embedding_dim, num_hid)

    # Variables used for early stopping
    best_valid_CE = np.infty
    end_training = False

    # Initialize the momentum vector to all zeros
    delta = Params.zeros(len(vocab), config['context_len'], embedding_dim, num_hid)

    this_chunk_CE = 0.
    batch_count = 0
    for epoch in range(1, config['epochs'] + 1):
        if end_training:
            break
        
        print
        print 'Epoch', epoch

        for m, (input_batch, target_batch) in enumerate(get_batches(train_inputs, train_targets,
                                                                    config['batch_size'])):
            batch_count += 1
            
            # Forward propagate
            activations = model.compute_activations(input_batch)

            # Compute loss derivative
            expanded_target_batch = model.indicator_matrix(target_batch)
            loss_derivative = model.compute_loss_derivative(activations.output_layer, expanded_target_batch)
            loss_derivative /= config['batch_size']

            # Measure loss function
            cross_entropy = model.compute_loss(activations.output_layer, expanded_target_batch) / config['batch_size']
            this_chunk_CE += cross_entropy
            if batch_count % config['show_training_CE_after'] == 0:
                print 'Batch {} Train CE {:1.3f}'.format(
                    batch_count, this_chunk_CE / config['show_training_CE_after'])
                this_chunk_CE = 0.
            
            # Backpropagate
            loss_gradient = model.back_propagate(input_batch, activations, loss_derivative)

            # Update the momentum vector and model parameters
            delta = config['momentum'] * delta + loss_gradient
            model.params -= config['learning_rate'] * delta

            # Validate
            if batch_count % config['show_validation_CE_after'] == 0:
                print 'Running validation...'
                cross_entropy = model.evaluate(valid_inputs, valid_targets)
                print 'Validation cross-entropy: {:1.3f}'.format(cross_entropy)

                if cross_entropy > best_valid_CE:
                    print 'Validation error increasing!  Training stopped.'
                    end_training = True
                    break
                    
                best_valid_CE = cross_entropy
            
            
    print
    train_CE = model.evaluate(train_inputs, train_targets)
    print 'Final training cross-entropy: {:1.3f}'.format(train_CE)
    valid_CE = model.evaluate(valid_inputs, valid_targets)
    print 'Final validation cross-entropy: {:1.3f}'.format(valid_CE)
    test_CE = model.evaluate(test_inputs, test_targets)
    print 'Final test cross-entropy: {:1.3f}'.format(test_CE)

    return model


            
            


    

    

    


    
