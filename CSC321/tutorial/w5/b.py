from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat



M = loadmat("mnist_all.mat")

def get_test(M):
    batch_xs = np.zeros((0, 28*28))
    batch_y_s = np.zeros( (0, 10))
    
    test_k =  ["test"+str(i) for i in range(10)]
    for k in range(10):
        batch_xs = np.vstack((batch_xs, ((np.array(M[test_k[k]])[:])/255.)  ))
        one_hot = np.zeros(10)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s,   np.tile(one_hot, (len(M[test_k[k]]), 1))   ))
    return batch_xs, batch_y_s


def get_train(M):
    batch_xs = np.zeros((0, 28*28))
    batch_y_s = np.zeros( (0, 10))
    
    train_k =  ["train"+str(i) for i in range(10)]
    for k in range(10):
        batch_xs = np.vstack((batch_xs, ((np.array(M[train_k[k]])[:])/255.)  ))
        one_hot = np.zeros(10)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s,   np.tile(one_hot, (len(M[train_k[k]]), 1))   ))
    return batch_xs, batch_y_s
        

train_x, train_y = get_train(M)
test_x, test_y = get_test(M)

dim_x = 28*28  # input layer
dim_h = 20     # hidden layer
dim_out = 10   # output layer


dtype_float = torch.FloatTensor

# subsample
train_idx = np.random.permutation(range(train_x.shape[0]))[:1000]
x = Variable(torch.from_numpy(train_x[train_idx]), requires_grad=False).type(dtype_float)
y = Variable(torch.from_numpy(train_y[train_idx].astype(float)), requires_grad=False).type(dtype_float)

b0 = Variable(torch.randn((1, dim_h)), requires_grad=True)
W0 = Variable(torch.randn((dim_x, dim_h)), requires_grad=True)  # from -> to
b1 = Variable(torch.randn((1, dim_out)), requires_grad=True)
W1 = Variable(torch.randn((dim_h, dim_out)), requires_grad=True)

# defines the computation graph
def model(x, b0, W0, b1, W1):
    h = torch.nn.ReLU()(torch.matmul(x, W0) + b0.repeat(x.data.shape[0], 1))
    out = torch.matmul(h, W1) + b1.repeat(h.data.shape[0], 1)
    return out


y_out = model(x, b0, W0, b1, W1)

# loss function
logSoftMax = torch.nn.LogSoftmax()

learning_rate = 1e-1

for t in range(1000):
    # compute output of model on training data
    y_out = model(x, b0, W0, b1, W1)
    loss = -torch.mean(torch.sum(y * logSoftMax(y_out), 1))
    loss.backward()
    # gradient descent update
    b0.data -= learning_rate * b0.grad.data
    W0.data -= learning_rate * W0.grad.data
    b1.data -= learning_rate * b1.grad.data
    W1.data -= learning_rate * W1.grad.data
    
    # clear out stored gradient data
    b0.grad.data.zero_()
    W0.grad.data.zero_()
    b1.grad.data.zero_()
    W1.grad.data.zero_()
    
    #print(loss.data.numpy())

x_test_all_var = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
y_test_out = model(x_test_all_var, b0, W0, b1, W1).data.numpy()