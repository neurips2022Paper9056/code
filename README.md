# CODE
This repository contains the source code and trained networks for paper number 9056.
Each experiment is self-contained under the project folder (therefore, there is redundant code between these).
The readme in the respective folders describe how to use the code.


## Installation
These instructions have been tested on Ubuntu 20.04 only.

#### Prerequisites
* NVIDIA GPU
* CUDA 11.3 (probably another version will work, too)
* cuDNN 8.2.1 (or what fits to your CUDA installation)
* anaconda
* If you want to train the state representations, 128 GB of system RAM are required and the GPU should have at least 16 GB of RAM, 24 GB or more are recommended, otherwise, training is too slow (we provide a parameter with which the GPU memory requirement can be traded off with training speed)

This installation guide assumes that everything is installed under `$HOME/git`

#### Ubuntu packages
```
sudo apt-get install libglew-dev freeglut3-dev liblapack-dev libf2c2-dev gnupg libjsoncpp-dev libx11-dev libeigen3-dev libassimp-dev libqhull-dev libglfw3-dev libann-dev libhdf5-dev
```

If you want to run the code on a headless server, install
```
sudo apt install xserver-xorg-video-dummy
```
and check https://techoverflow.net/2019/02/23/how-to-run-x-server-using-xserver-xorg-video-dummy-driver-on-ubuntu/ how to use it.


#### Create an anaconda environment
```
conda create --name NeRF-RL python=3.8
```

The following assumes that the conda environment `NeRF-RL` is activated for the rest of the installation and also when running the code
```
conda activate NeRF-RL
```

#### Conda packages
```
conda install h5py matplotlib
```

#### Pip packages
```
conda install stable_baselines3
```

#### Pytorch form source
Basically follow the instructions from https://github.com/pytorch/pytorch#from-source

```
conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
```
Adjust the following to your CUDA version
```
conda install -c pytorch magma-cuda113

```

Clone the repo and install (can take some time)
```
cd $HOME/git
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```


#### Bullet from source
```
cd $HOME/git
git clone https://github.com/bulletphysics/bullet3.git
cd bullet3
git checkout -b 2.88
mkdir -p build_cmake
cd build_cmake
cmake \
 -DCMAKE_INSTALL_PREFIX=$HOME/git/bulletLib \
 -DBUILD_PYBULLET=OFF \
 -DBUILD_PYBULLET_NUMPY=OFF \
 -DUSE_DOUBLE_PRECISION=ON \
 -DBT_USE_EGL=ON \
 -DCMAKE_BUILD_TYPE=Release \
 ..
make -j 4
make install
```
