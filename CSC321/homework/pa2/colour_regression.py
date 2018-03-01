"""
Colourization framed as a regression problem.
"""

from __future__ import print_function
import argparse
import os
import numpy as np
import numpy.random as npr
import scipy.misc
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from load_data import load_cifar10

from colourization import process, get_batch, MyConv2d

class RegressionCNN(nn.Module):
    def __init__(self, kernel, num_filters):
        # first call parent's initialization function
        super(RegressionCNN, self).__init__()
        padding = kernel // 2 
        # ensures that the input volume and output volume will have the same size spatially.
        
        self.downconv1 = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU())
        self.downconv2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters*2, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU())

        self.rfconv = nn.Sequential(
            nn.Conv2d(num_filters*2, num_filters*2, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU())

        self.upconv1 = nn.Sequential(
            nn.Conv2d(num_filters*2, num_filters, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU())
        self.upconv2 = nn.Sequential(
            nn.Conv2d(num_filters, 3, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(3),
            nn.ReLU())

        self.finalconv = MyConv2d(3, 3, kernel_size=kernel)

    def forward(self, x):
        out = self.downconv1(x)
        out = self.downconv2(out)
        out = self.rfconv(out)
        out = self.upconv1(out)
        out = self.upconv2(out)
        out = self.finalconv(out)
        return out

######################################################################
# Training
######################################################################

def get_torch_vars(xs, ys, gpu=False):
    """
    Helper function to convert numpy arrays to pytorch tensors.
    If GPU is used, move the tensors to GPU.

    Args:
      xs (float numpy tenosor): greyscale input
      ys (float numpy tenosor): colour output
      gpu (bool): whether to move pytorch tensor to GPU
    Returns:
      Variable(xs), Variable(ys)
    """
    xs = torch.from_numpy(xs).float()
    ys = torch.from_numpy(ys).float()
    if gpu:
        xs = xs.cuda()
        ys = ys.cuda()
    return Variable(xs), Variable(ys)

def train(cnn, epochs=80, learn_rate=0.001, batch_size=100, gpu=True):
    """
    Train a regression CNN. Note that you do not need this function.
    Included for refrence.
    """
    if gpu:
        cnn.cuda()

    # Set up L2 loss
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learn_rate)

    # Loading & transforming data
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    train_rgb, train_grey = process(x_train, y_train)
    test_rgb, test_grey = process(x_test, y_test)

    print("Beginning training ...")

    for epoch in range(epochs):
        # Train the Model
        cnn.train() # Change model to 'train' mode
        for i, (xs, ys) in enumerate(get_batch(train_grey,
                                               train_rgb,
                                               batch_size)):
            images, labels = get_torch_vars(xs, ys, gpu)
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, epochs, loss.data[0]))

        # Evaluate the model
        cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        losses = []
        for i, (xs, ys) in enumerate(get_batch(test_grey,
                                               test_rgb,
                                               batch_size)):
            images, labels = get_torch_vars(xs, ys, gpu)
            outputs = cnn(images)

            val_loss = criterion(outputs, labels)
            losses.append(val_loss.data[0])

        val_loss = np.mean(losses)
        print('Epoch [%d/%d], Val Loss: %.4f' % (epoch+1, epochs, val_loss))

    # Save the Trained Model
    torch.save(cnn.state_dict(), 'regression_cnn_k%d_f%d.pkl' % (
        args.kernel, args.num_filters))

def plot(grey, gtcolour, predcolour, path):
    """
    Plot input, gt output and predicted output as an image.

    Args:
      grey: numpy tenosor of shape Nx1xHxW
      gtcolour: numpy tenosor of shape Nx3xHxW
      predcolour: numpy tenosor of shape Nx3xHxW
      path: path to save the image
    """
    grey = np.transpose(grey, [0,2,3,1])
    gtcolour = np.transpose(gtcolour, [0,2,3,1])
    predcolour = np.transpose(predcolour, [0,2,3,1])

    img = np.vstack([
      np.hstack(np.tile(grey, [1,1,1,3])),
      np.hstack(gtcolour),
      np.hstack(predcolour)])
    scipy.misc.toimage(img, cmin=0, cmax=1).save(path)

######################################################################
# MAIN
######################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train colourization")
    parser.add_argument('--gpu', action='store_true', default=False,
                        help="Use GPU for training")
    parser.add_argument('-k', '--kernel', default=3,
                        help="Convolution kernel size")
    parser.add_argument('-f', '--num_filters', default=32,
                        help="Base number of convolution filters")
    args = parser.parse_args()

    npr.seed(0)

    cnn = RegressionCNN(args.kernel, args.num_filters)

    # Uncomment to train. You do not need this for the assignment.
    # Included for completeness
    #train(cnn); exit(0)

    print("Loading weights...")
    checkpoint = torch.load('weights/regression_cnn_k%d_f%d.pkl' % (args.kernel, args.num_filters), map_location=lambda storage, loc: storage)
    cnn.load_state_dict(checkpoint)

    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    test_rgb, test_grey = process(x_test, y_test)

    # Create output folder if not created
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    
    print("Generating predictions...")
    grey = test_grey[:15]
    gtrgb = test_rgb[:15]
    predrgb = cnn(Variable(torch.from_numpy(grey).float()))
    predrgb = predrgb.data.numpy() 
    plot(grey, gtrgb, predrgb, "outputs/regression_output.png")

