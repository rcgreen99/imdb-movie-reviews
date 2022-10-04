# Provisions



## Pytorch/CUDA/CudNN

The correct version of torch, CUDA, and cuDNN must be installed to run torch with device="cuda"
The best way to do this that I've found is this:

1. Remove all nvidia drivers or programs
2. Auto install apt-get suggested nvidia-driver
3. reboot
4. Download and install nvidia-cuda-toolkit w/o driver
5. Downlaod and install cuDNN
6. If problems here, make sure correct torch version is installed

## Docker

To get docker to work on Ubuntu 22.04 with GPU support I had to do a couple of extra things on top of installing nvidia-cuda-toolkit locally.

1. Install nvidia-containers-toolkit
2. Use ```--gpus all``` flag when running docker container (docker run ...)
