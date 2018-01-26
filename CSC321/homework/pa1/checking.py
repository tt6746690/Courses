import cPickle
import numpy as np

import language_model


PARTIALLY_TRAINED_MODEL = 'partially_trained.pk'

EPS = 1e-4

def relative_error(a, b):
    return np.abs(a - b) / (np.abs(a) + np.abs(b))


def check_output_derivatives(model, input_batch, target_batch):
    def softmax(z):
        z = z.copy()
        z -= z.max(1).reshape((-1, 1))
        y = np.exp(z)
        y /= y.sum(1).reshape((-1, 1))
        return y
    
    batch_size = input_batch.shape[0]
    z = np.random.normal(size=(batch_size, model.vocab_size))
    y = softmax(z)

    expanded_target_batch = model.indicator_matrix(target_batch)
    loss_derivative = model.compute_loss_derivative(y, expanded_target_batch)

    if loss_derivative is None:
        print 'Loss derivative not implemented yet.'
        return False

    if loss_derivative.shape != (batch_size, model.vocab_size):
        print 'Loss derivative should be size {} but is actually {}.'.format(
            (batch_size, model.vocab_size), loss_derivative.shape)
        return False

    def obj(z):
        y = softmax(z)
        return model.compute_loss(y, expanded_target_batch)

    for count in range(1000):
        i, j = np.random.randint(0, loss_derivative.shape[0]), np.random.randint(0, loss_derivative.shape[1])

        z_plus = z.copy()
        z_plus[i, j] += EPS
        obj_plus = obj(z_plus)

        z_minus = z.copy()
        z_minus[i, j] -= EPS
        obj_minus = obj(z_minus)

        empirical = (obj_plus - obj_minus) / (2. * EPS)
        rel = relative_error(empirical, loss_derivative[i, j])
        if rel > 1e-4:
            print 'The loss derivative has a relative error of {}, which is too large.'.format(rel)
            return False

    print 'The loss derivative looks OK.'
    return True


def check_param_gradient(model, param_name, input_batch, target_batch):
    activations = model.compute_activations(input_batch)
    expanded_target_batch = model.indicator_matrix(target_batch)
    loss_derivative = model.compute_loss_derivative(activations.output_layer, expanded_target_batch)
    param_gradient = model.back_propagate(input_batch, activations, loss_derivative)

    def obj(model):
        activations = model.compute_activations(input_batch)
        return model.compute_loss(activations.output_layer, expanded_target_batch)

    dims = getattr(model.params, param_name).shape
    is_matrix = (len(dims) == 2)

    if getattr(param_gradient, param_name).shape != dims:
        print 'The gradient for {} should be size {} but is actually {}.'.format(
            param_name, dims, getattr(param_gradient, param_name).shape)
        return

    for count in range(1000):
        if is_matrix:
            slc = np.random.randint(0, dims[0]), np.random.randint(0, dims[1])
        else:
            slc = np.random.randint(dims[0])

        model_plus = model.copy()
        getattr(model_plus.params, param_name)[slc] += EPS
        obj_plus = obj(model_plus)

        model_minus = model.copy()
        getattr(model_minus.params, param_name)[slc] -= EPS
        obj_minus = obj(model_minus)
        
        empirical = (obj_plus - obj_minus) / (2. * EPS)
        exact = getattr(param_gradient, param_name)[slc]
        rel = relative_error(empirical, exact)
        if rel > 1e-4:
            print 'The loss derivative has a relative error of {}, which is too large.'.format(rel)
            return False

    print 'The gradient for {} looks OK.'.format(param_name)
    
    

def load_partially_trained_model():
    obj = cPickle.load(open(PARTIALLY_TRAINED_MODEL, 'rb'))
    params = language_model.Params(obj['word_embedding_weights'], obj['embed_to_hid_weights'],
                                   obj['hid_to_output_weights'], obj['hid_bias'],
                                   obj['output_bias'])
    vocab = obj['vocab']
    return language_model.Model(params, vocab)


def check_gradients():
    """Check the computed gradients using finite differences."""
    np.random.seed(0)

    np.seterr(all='ignore')   # suppress a warning which is harmless
    
    model = load_partially_trained_model()
    data_obj = cPickle.load(open('data.pk', 'rb'))
    train_inputs, train_targets = data_obj['train_inputs'], data_obj['train_targets']
    input_batch = train_inputs[:100, :]
    target_batch = train_targets[:100]

    if not check_output_derivatives(model, input_batch, target_batch):
        return

    for param_name in ['word_embedding_weights', 'embed_to_hid_weights', 'hid_to_output_weights',
                       'hid_bias', 'output_bias']:
        check_param_gradient(model, param_name, input_batch, target_batch)




def print_gradients():
    """Print out certain derivatives for grading."""
    
    model = load_partially_trained_model()
    data_obj = cPickle.load(open('data.pk', 'rb'))
    train_inputs, train_targets = data_obj['train_inputs'], data_obj['train_targets']
    input_batch = train_inputs[:100, :]
    target_batch = train_targets[:100]

    activations = model.compute_activations(input_batch)
    expanded_target_batch = model.indicator_matrix(target_batch)
    loss_derivative = model.compute_loss_derivative(activations.output_layer, expanded_target_batch)
    param_gradient = model.back_propagate(input_batch, activations, loss_derivative)

    print 'loss_derivative[2, 5]', loss_derivative[2, 5]
    print 'loss_derivative[2, 121]', loss_derivative[2, 121]
    print 'loss_derivative[5, 33]', loss_derivative[5, 33]
    print 'loss_derivative[5, 31]', loss_derivative[5, 31]
    print
    print 'param_gradient.word_embedding_weights[27, 2]', param_gradient.word_embedding_weights[27, 2]
    print 'param_gradient.word_embedding_weights[43, 3]', param_gradient.word_embedding_weights[43, 3]
    print 'param_gradient.word_embedding_weights[22, 4]', param_gradient.word_embedding_weights[22, 4]
    print 'param_gradient.word_embedding_weights[2, 5]', param_gradient.word_embedding_weights[2, 5]
    print
    print 'param_gradient.embed_to_hid_weights[10, 2]', param_gradient.embed_to_hid_weights[10, 2]
    print 'param_gradient.embed_to_hid_weights[15, 3]', param_gradient.embed_to_hid_weights[15, 3]
    print 'param_gradient.embed_to_hid_weights[30, 9]', param_gradient.embed_to_hid_weights[30, 9]
    print 'param_gradient.embed_to_hid_weights[35, 21]', param_gradient.embed_to_hid_weights[35, 21]
    print
    print 'param_gradient.hid_bias[10]', param_gradient.hid_bias[10]
    print 'param_gradient.hid_bias[20]', param_gradient.hid_bias[20]
    print
    print 'param_gradient.output_bias[0]', param_gradient.output_bias[0]
    print 'param_gradient.output_bias[1]', param_gradient.output_bias[1]
    print 'param_gradient.output_bias[2]', param_gradient.output_bias[2]
    print 'param_gradient.output_bias[3]', param_gradient.output_bias[3]
    





    

    
    
    

    
