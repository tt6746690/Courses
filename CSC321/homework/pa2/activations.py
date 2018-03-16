import argparse
import numpy as np
import numpy.random as npr
import os
import scipy.misc
import torch 
import torch.nn as nn

from colourization import (get_torch_vars, get_cat_rgb, get_rgb_cat,
                           process, UNet, CNN, DilatedUNet)
from load_data import load_cifar10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train colourization")
    parser.add_argument('index', default=0, type=int,
                        help="Image index to plot")
    parser.add_argument('--checkpoint', default="",
                        help="Model file to load and save")
    parser.add_argument('--outdir', default="outputs/act",
                        help="Directory to save the file")
    parser.add_argument('-m', '--model', choices=["CNN", "UNet", "DUNet"],
                        help="Model to run")
    parser.add_argument('-k', '--kernel', default=3, type=int,
                        help="Convolution kernel size")
    parser.add_argument('-f', '--filters', default=32, type=int,
                        help="Base number of convolution filters")
    parser.add_argument('-c', '--colours',
                        default='colours/colour_kmeans24_cat7.npy',
                        help="Discrete colour clusters to use")
    args = parser.parse_args()

    # LOAD THE COLOURS CATEGORIES
    colours = np.load(args.colours)[0]
    num_colours = np.shape(colours)[0]

    # Load the data first for consistency
    print("Loading data...")
    npr.seed(0)
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    test_rgb, test_grey = process(x_test, y_test)
    test_rgb_cat = get_rgb_cat(test_rgb, colours)

    # LOAD THE MODEL
    if args.model == "CNN":
        cnn = CNN(args.kernel, args.filters, num_colours)
    elif args.model == "UNet":
        cnn = UNet(args.kernel, args.filters, num_colours)
    else: # model == "DUNet":
        cnn = DilatedUNet(args.kernel, args.filters, num_colours)

    print("Loading checkpoint...")
    cnn.load_state_dict(torch.load(args.checkpoint, map_location=lambda storage, loc: storage))

    # Take the idnex of the test image
    id = args.index
    outdir = args.outdir + str(id)
    os.mkdir(outdir)
    images, labels = get_torch_vars(np.expand_dims(test_grey[id], 0),
                                    np.expand_dims(test_rgb_cat[id], 0))
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1, keepdim=True)
    predcolor = get_cat_rgb(predicted.cpu().numpy()[0,0,:,:], colours)
    scipy.misc.toimage(predcolor, cmin=0, cmax=1) \
            .save(os.path.join(outdir, "filter_output_%d.png" % id))


    scipy.misc.toimage(np.transpose(test_rgb[id], [1,2,0]), cmin=0, cmax=1) \
            .save(os.path.join(outdir, "filter_input_rgb_%d.png" % id))
    scipy.misc.toimage(test_grey[id,0,:,:], cmin=0, cmax=1) \
            .save(os.path.join(outdir, "filter_input_%d.png" % id))

    def add_border(img):
        return np.pad(img, 1, "constant", constant_values=1.0)

    def draw_activations(path, activation, imgwidth=4):
        img = np.vstack([
            np.hstack([
                add_border(filter) for filter in
                activation[i*imgwidth:(i+1)*imgwidth,:,:]])
            for i in range(activation.shape[0] // imgwidth)])
        scipy.misc.imsave(path, img)

    for i, tensor in enumerate([cnn.out1, cnn.out2, cnn.out3, cnn.out4, cnn.out5]):
        draw_activations(
            os.path.join(outdir, "filter_out%d_%d.png" % (i, id)),
            tensor.data.cpu().numpy()[0])
