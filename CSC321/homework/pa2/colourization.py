"""
Colourization of CIFAR-10 Horses via classification.
"""

from __future__ import print_function
import argparse
import os
import math
import numpy as np
import numpy.random as npr
import scipy.misc
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg') # switch backend
import matplotlib.pyplot as plt 


from load_data import load_cifar10

HORSE_CATEGORY = 7

######################################################################
# Data related code
######################################################################
def get_rgb_cat(xs, colours):
    """
    Get colour categories given RGB values. This function doesn't
    actually do the work, instead it splits the work into smaller
    chunks that can fit into memory, and calls helper function
    _get_rgb_cat

    Args:
      xs: float numpy array of RGB images in [B, C, H, W] format
      colours: numpy array of colour categories and their RGB values
    Returns:
      result: int numpy array of shape [B, 1, H, W]
    """
    if np.shape(xs)[0] < 100:
        return _get_rgb_cat(xs)
    batch_size = 100
    nexts = []
    for i in range(0, np.shape(xs)[0], batch_size):
        next = _get_rgb_cat(xs[i:i+batch_size,:,:,:], colours)
        nexts.append(next)
    result = np.concatenate(nexts, axis=0)
    return result

def _get_rgb_cat(xs, colours):
    """
    Get colour categories given RGB values. This is done by choosing
    the colour in `colours` that is the closest (in RGB space) to
    each point in the image `xs`. This function is a little memory
    intensive, and so the size of `xs` should not be too large.

    Args:
      xs: float numpy array of RGB images in [B, C, H, W] format
      colours: numpy array of colour categories and their RGB values
    Returns:
      result: int numpy array of shape [B, 1, H, W]
    """
    num_colours = np.shape(colours)[0]
    xs = np.expand_dims(xs, 0)
    cs = np.reshape(colours, [num_colours,1,3,1,1])
    dists = np.linalg.norm(xs-cs, axis=2) # 2 = colour axis
    cat = np.argmin(dists, axis=0)
    cat = np.expand_dims(cat, axis=1)
    return cat

def get_cat_rgb(cats, colours):
    """
    Get RGB colours given the colour categories

    Args:
      cats: integer numpy array of colour categories
      colours: numpy array of colour categories and their RGB values
    Returns:
      numpy tensor of RGB colours
    """
    return colours[cats]

def process(xs, ys, max_pixel=256.0):
    """
    Pre-process CIFAR10 images by taking only the horse category,
    shuffling, and have colour values be bound between 0 and 1

    Args:
      xs: the colour RGB pixel values
      ys: the category labels
      max_pixel: maximum pixel value in the original data
    Returns:
      xs: value normalized and shuffled colour images
      grey: greyscale images, also normalized so values are between 0 and 1
    """
    xs = xs / max_pixel
    xs = xs[np.where(ys == HORSE_CATEGORY)[0], :, :, :]
    npr.shuffle(xs)
    grey = np.mean(xs, axis=1, keepdims=True)
    return (xs, grey)

def get_batch(x, y, batch_size):
    '''
    Generated that yields batches of data

    Args:
      x: input values
      y: output values
      batch_size: size of each batch
    Yields:
      batch_x: a batch of inputs of size at most batch_size
      batch_y: a batch of outputs of size at most batch_size
    '''
    N = np.shape(x)[0]
    assert N == np.shape(y)[0]
    for i in range(0, N, batch_size):
        batch_x = x[i:i+batch_size, :,:,:]
        batch_y = y[i:i+batch_size, :,:,:]
        yield (batch_x, batch_y)

def plot(input, gtlabel, output, colours, path):
    """
    Generate png plots of input, ground truth, and outputs

    Args:
      input: the greyscale input to the colourization CNN
      gtlabel: the grouth truth categories for each pixel
      output: the predicted categories for each pixel
      colours: numpy array of colour categories and their RGB values
      path: output path
    """
    grey = np.transpose(input[:10,:,:,:], [0,2,3,1])
    gtcolor = get_cat_rgb(gtlabel[:10,0,:,:], colours)
    predcolor = get_cat_rgb(output[:10,0,:,:], colours)

    img = np.vstack([
      np.hstack(np.tile(grey, [1,1,1,3])),
      np.hstack(gtcolor),
      np.hstack(predcolor)])
    scipy.misc.toimage(img, cmin=0, cmax=1).save(path)


######################################################################
# MODELS
######################################################################

class MyConv2d(nn.Module):
    """
    Our simplified implemented of nn.Conv2d module for 2D convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=None):
        super(MyConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if padding is None:
            self.padding = kernel_size // 2
        else:
            self.padding = padding
        
        # register optimizable parameters to class instances 
        # for each filter (out_channels number of them):
        #     weights: in_channels x kernel_size x kernel_size 
        #     bias: 1
        # order of weight matrix necessary as args to F.conv2d()
        self.weight = nn.parameter.Parameter(torch.Tensor(
            out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.parameter.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n)
        # fills tensor with numbers sampled from uniform distribution
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, padding=self.padding)

class MyDilatedConv2d(MyConv2d):
    """
    Dilated Convolution 2D
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(MyDilatedConv2d, self).__init__(in_channels,
                                              out_channels,
                                              kernel_size)
        self.dilation = dilation

    def forward(self, input):
        ############### YOUR CODE GOES HERE ############### 

        # pad weight matrix replicatively based on dilation specified
        replicative_padding = (self.dilation, self.dilation, self.dilation, self.dilation)
        self.dilated_weight = torch.nn.functional.pad(self.weight, replicative_padding, "replicate")

        # zero the padded entries
        for i in range(self.kernel_size - 1):
            j = (self.dilation + 1) * i + 1
            self.dilated_weight.data[:,:,j:j+self.dilation,:] = 0
            self.dilated_weight.data[:,:,:,j:j+self.dilation] = 0

        # update kernel size such that C_out is equal to C_in
        receptive_field_size = (self.kernel_size-1)*self.dilation + self.kernel_size

        return F.conv2d(input, self.dilated_weight, self.bias, padding=receptive_field_size//2)
        ###################################################

class CNN(nn.Module):
    def __init__(self, kernel, num_filters, num_colours):
        super(CNN, self).__init__()
        padding = kernel // 2

        ############### YOUR CODE GOES HERE ############### 
        self.downconv1 = nn.Sequential(
            MyConv2d(1, num_filters, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU())
        self.downconv2 = nn.Sequential(
            MyConv2d(num_filters, num_filters*2, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU())

        self.rfconv = nn.Sequential(
            MyConv2d(num_filters*2, num_filters*2, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU())

        self.upconv1 = nn.Sequential(
            MyConv2d(num_filters*2, num_filters, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU())
        self.upconv2 = nn.Sequential(
            MyConv2d(num_filters, num_colours, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_colours),
            nn.ReLU())

        self.finalconv = MyConv2d(num_colours, num_colours, kernel_size=kernel)
        ###################################################

    def forward(self, x):
        self.out1 = self.downconv1(x)
        self.out2 = self.downconv2(self.out1)
        self.out3 = self.rfconv(self.out2)
        self.out4 = self.upconv1(self.out3)
        self.out5 = self.upconv2(self.out4)
        self.out_final = self.finalconv(self.out5)
        return self.out_final

class UNet(nn.Module):
    def __init__(self, kernel, num_filters, num_colours):
        super(UNet, self).__init__()

        ############### YOUR CODE GOES HERE ############### 
        padding = kernel // 2 
        
        self.downconv1 = nn.Sequential(
            MyConv2d(1, num_filters, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU())
        self.downconv2 = nn.Sequential(
            MyConv2d(num_filters, num_filters*2, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU())

        self.rfconv = nn.Sequential(
            MyConv2d(num_filters*2, num_filters*2, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU())

        self.upconv1 = nn.Sequential(
            MyConv2d(num_filters*4, num_filters, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU())
        self.upconv2 = nn.Sequential(
            MyConv2d(num_filters*2, num_colours, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_colours),
            nn.ReLU())

        self.finalconv = MyConv2d(num_colours+1, num_colours, kernel_size=kernel)
        ###################################################

    def forward(self, x):
        # x:     (N, 1, H, W)
        # out1:  (N, num_filters, H/2, W/2)
        # out2:  (N, 2*num_filters, H/4, W/4)
        # out3:  (N, 2*num_filters, H/4, W/4)
        # out4:  (N, num_filters, H/2, W/2)
        # out5:  (N, num_colors, H, W)
        # out_final: (N, num_colors, H, W)
        ############### YOUR CODE GOES HERE ############### 
        self.out1 = self.downconv1(x)
        self.out2 = self.downconv2(self.out1)
        self.out3 = self.rfconv(self.out2)
        self.out4 = self.upconv1(torch.cat((self.out3, self.out2), 1))
        self.out5 = self.upconv2(torch.cat((self.out4, self.out1), 1))
        self.out_final = self.finalconv(torch.cat((self.out5, x), 1))
        return self.out_final
        ###################################################

class DilatedUNet(UNet):
    def __init__(self, kernel, num_filters, num_colours):
        super(DilatedUNet, self).__init__(kernel, num_filters, num_colours)
        # replace the intermediate dilations
        self.rfconv = nn.Sequential(
            MyDilatedConv2d(num_filters*2, num_filters*2, kernel_size=kernel, dilation=1),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU())

######################################################################
# Torch Helper
######################################################################

def get_torch_vars(xs, ys, gpu=False):
    """
    Helper function to convert numpy arrays to pytorch tensors.
    If GPU is used, move the tensors to GPU.

    Args:
      xs (float numpy tenosor): greyscale input
      ys (int numpy tenosor): categorical labels 
      gpu (bool): whether to move pytorch tensor to GPU
    Returns:
      Variable(xs), Variable(ys)
    """
    xs = torch.from_numpy(xs).float()
    ys = torch.from_numpy(ys).long()
    if gpu:
        xs = xs.cuda()
        ys = ys.cuda()
    return Variable(xs), Variable(ys)

def compute_loss(criterion, outputs, labels, batch_size, num_colours):
    """
    Helper function to compute the loss. Since this is a pixelwise
    prediction task we need to reshape the output and ground truth
    tensors into a 2D tensor before passing it in to the loss criteron.

    Args:
      criterion: pytorch loss criterion
      outputs (pytorch tensor): predicted labels from the model
      labels (pytorch tensor): ground truth labels
      batch_size (int): batch size used for training
      num_colours (int): number of colour categories
    Returns:
      pytorch tensor for loss
    """

    loss_out = outputs.transpose(1,3) \
                      .contiguous() \
                      .view([batch_size*32*32, num_colours])
    loss_lab = labels.transpose(1,3) \
                      .contiguous() \
                      .view([batch_size*32*32])
    return criterion(loss_out, loss_lab)

def run_validation_step(cnn, criterion, test_grey, test_rgb_cat, batch_size,
                        colour, plotpath=None):
    correct = 0.0
    total = 0.0
    losses = []
    for i, (xs, ys) in enumerate(get_batch(test_grey,
                                           test_rgb_cat,
                                           batch_size)):
        images, labels = get_torch_vars(xs, ys, args.gpu)
        outputs = cnn(images)

        val_loss = compute_loss(criterion,
                                outputs,
                                labels,
                                batch_size=args.batch_size,
                                num_colours=num_colours)
        losses.append(val_loss.data[0])

        _, predicted = torch.max(outputs.data, 1, keepdim=True)
        total += labels.size(0) * 32 * 32
        correct += (predicted == labels.data).sum()

    if plotpath: # only plot if a path is provided
        plot(xs, ys, predicted.cpu().numpy(), colours, plotpath)

    val_loss = np.mean(losses)
    val_acc = 100 * correct / total
    return val_loss, val_acc


######################################################################
# MAIN
######################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train colourization")
    parser.add_argument('--gpu', action='store_true', default=False,
                        help="Use GPU for training")
    parser.add_argument('--valid', action="store_true", default=False,
                        help="Perform validation only (don't train)")
    parser.add_argument('--checkpoint', default="",
                        help="Model file to load and save")
    parser.add_argument('--plot', action="store_true", default=False,
                        help="Plot outputs every epoch during training")
    parser.add_argument('-c', '--colours',
                        default='colours/colour_kmeans24_cat7.npy',
                        help="Discrete colour clusters to use")
    parser.add_argument('-m', '--model', choices=["CNN", "UNet", "DUNet"],
                        help="Model to run")
    parser.add_argument('-k', '--kernel', default=3, type=int,
                        help="Convolution kernel size")
    parser.add_argument('-f', '--num_filters', default=32, type=int,
                        help="Base number of convolution filters")
    parser.add_argument('-l', '--learn_rate', default=0.001, type=float,
                        help="Learning rate")
    parser.add_argument('-b', '--batch_size', default=100, type=int,
                        help="Batch size")
    parser.add_argument('-e', '--epochs', default=25, type=int,
                        help="Number of epochs to train")
    parser.add_argument('-s', '--seed', default=0, type=int,
                        help="Numpy random seed")

    args = parser.parse_args()

    # Set the maximum number of threads to prevent crash in Teaching Labs
    torch.set_num_threads(5)

    # Numpy random seed
    npr.seed(args.seed)

    # LOAD THE COLOURS CATEGORIES
    colours = np.load(args.colours)[0]
    num_colours = np.shape(colours)[0]

    # LOAD THE MODEL
    if args.model == "CNN":
        cnn = CNN(args.kernel, args.num_filters, num_colours)
    elif args.model == "UNet":
        cnn = UNet(args.kernel, args.num_filters, num_colours)
    else: # model == "DUNet":
        cnn = DilatedUNet(args.kernel, args.num_filters, num_colours)

    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=args.learn_rate)

    # DATA
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    print("Transforming data...")
    train_rgb, train_grey = process(x_train, y_train)
    train_rgb_cat = get_rgb_cat(train_rgb, colours)
    test_rgb, test_grey = process(x_test, y_test)
    test_rgb_cat = get_rgb_cat(test_rgb, colours)

    # Create the outputs folder if not created already
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    # Run validation only
    if args.valid:
        if not args.checkpoint:
            raise ValueError("You need to give trained model to evaluate")

        print("Loading checkpoint...")
        cnn.load_state_dict(torch.load(args.checkpoint, map_location=lambda storage, loc: storage))
        img_path = "outputs/eval_%s.png" % args.model
        val_loss, val_acc = run_validation_step(cnn,
                                                criterion,
                                                test_grey,
                                                test_rgb_cat,
                                                args.batch_size,
                                                colours,
                                                img_path)
        print('Evaluating Model %s: %s' % (args.model, args.checkpoint))
        print('Val Loss: %.4f, Val Acc: %.1f%%' % (val_loss, val_acc))
        print('Sample output available at: %s' % img_path)
        exit(0)

    print("Beginning training ...")
    if args.gpu: cnn.cuda()
    start = time.time()

    train_losses = []
    valid_losses = []
    valid_accs = []
    for epoch in range(args.epochs):
        # Train the Model
        cnn.train() # Change model to 'train' mode
        losses = []
        for i, (xs, ys) in enumerate(get_batch(train_grey,
                                               train_rgb_cat,
                                               args.batch_size)):
            images, labels = get_torch_vars(xs, ys, args.gpu)
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)

            loss = compute_loss(criterion,
                                outputs,
                                labels,
                                batch_size=args.batch_size,
                                num_colours=num_colours)
            loss.backward()
            optimizer.step()
            losses.append(loss.data[0])

        # plot training images
        if args.plot:
            _, predicted = torch.max(outputs.data, 1, keepdim=True)
            plot(xs, ys, predicted.cpu().numpy(), colours,
                 'outputs/train_%d.png' % epoch)

        # plot training images
        avg_loss = np.mean(losses)
        train_losses.append(avg_loss)
        time_elapsed = time.time() - start
        print('Epoch [%d/%d], Loss: %.4f, Time (s): %d' % (
            epoch+1, args.epochs, avg_loss, time_elapsed))

        # Evaluate the model
        cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

        outfile = None
        if args.plot:
            outfile = 'outputs/test_%d.png' % epoch

        val_loss, val_acc = run_validation_step(cnn,
                                                criterion,
                                                test_grey,
                                                test_rgb_cat,
                                                args.batch_size,
                                                colours,
                                                outfile)

        time_elapsed = time.time() - start
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)
        print('Epoch [%d/%d], Val Loss: %.4f, Val Acc: %.1f%%, Time(s): %d' % (
            epoch+1, args.epochs, val_loss, val_acc, time_elapsed))

    # Plot training curve
    plt.plot(train_losses, "ro-", label="Train")
    plt.plot(valid_losses, "go-", label="Validation")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.savefig("outputs/training_curve.png")

    if args.checkpoint:
        print('Saving model...')
        torch.save(cnn.state_dict(), args.checkpoint)
