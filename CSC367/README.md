

+ intro to CUDA https://zhuanlan.zhihu.com/p/34587739
+ cuda by example code  https://github.com/CodedK/CUDA-by-Example-source-code-for-the-book-s-examples-


```
# use NVIDIA GPU at `@scheduler.cs.toronto.edu`
#       https://support.cs.toronto.edu/systems/slurmbookable.html

# access to remote GPU
ssh wpq@scheduler.cs.toronto.edu

# node usage status
sinfo -No '%10N %10T %.4c %.8z %.6m %f'
# user id
squeue

# start interactive shell (gpunode{1,11} has cuda10, compatible with gcc6)
srun --partition=gpunodes --nodelist=gpunode11 --mail-type=ALL,TIME_LIMIT_90 --mail-user=wpq@cs.toronto.edu  --pty bash --login

# cuda installation under `/pkgs_local`, i.e. use nvcc
/pkgs_local/cuda-10.0/bin/nvcc -v

# get binary from https://github.com/cdr/sshcode to somewhere
ln -s sshcode /usr/local/bin

# use vscode server on remote (remote cannot be `scheduler`)
sshcode --skipsync wpq@comps0.cs.toronto.edu /h/96/wpq/github/
```
