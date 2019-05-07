
## Link: http://www.cs.toronto.edu/~yani/csc320/

## Setup

```
# python-opencv tutorial
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html

# conda env support in jupyter notebook
conda install nb_conda

# select conda env for jupyter notebook
conda create -n csc320env -f csc320env.yml
conda activate csc320env
jupyter notebook
```


## Todos


+ do more tutorial under https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html
    + especially computational photography > calibration 


## Questions

+ Dy,Dx reversed?
+ what is a better way to compute gradient
+ operations using `floating` number to prevent overflow .. but needs to reallocate np array 
    + what is good convention `/255` or `astype`
+ remember to use `ddepth` to prevent overflow