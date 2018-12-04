# 
Code for the paper A Flexible Model for Training Action Localization with Varying Levels of Supervision, NIPS 2018

Created by Jean-Baptiste Alayrac and Guilhem Chéron at INRIA, Paris.

### Introduction

The webpage for this project is available [here](https://www.di.ens.fr/willow/research/weakactionloc/). It contains link to the [paper](https://arxiv.org/abs/1806.11328), and other material about the work.
This code reproduces the results presented in Table 1 of the paper for our method.

### License

Our code is released under the MIT License (refer to the LICENSE file for details).

### Cite

If you find this code useful in your research, please, consider citing our paper:

>@InProceedings{actoraction18,
>         author  = {Ch\'eron, Guilhem and Alayrac, Jean-Baptiste and Laptev, Ivan and Schmid, Cordelia},
>         title   = {A Flexible Model for Training Action Localization with Varying Levels of Supervision},
>         booktitle = {Neural Information Processing Systems (NIPS)},
>        year    = {2018}
>         }

### Contents

  1. [Requirements](#requirements)
  2. [Running the code](#running)

### Requirements

We run the code under python 2.7 with the following dependencies:

* numpy
* tqdm
* [mosek](https://docs.mosek.com/8.1/pythonapi/install-interface.html) (for which you'll need a license)
* pickle
* scikit-learn
* scipy

### Running

1) Clone this repo and go to the generated folder
  ```Shell
  git clone https://github.com/jalayrac/weak_action_loc.git
  cd weak_action_loc
  ```

2) Download and unpack the preprocessed features needed for the desired experiment in the data folder:

* UCF101-24 (11 GiB)
  ```Shell
  mkdir -p data
  cd data
  wget https://www.di.ens.fr/willow/research/weakactionloc/UCF101-24.tar.gz
  tar -xzvf UCF101-24.tar.gz
  cd ..
  ```

* DALY (40 GiB)
```Shell
  mkdir -p data
  cd data
  wget https://www.di.ens.fr/willow/research/weakactionloc/DALY.tar.gz
  tar -xzvf DALY.tar.gz
  cd ..
```

3) To obtain the results, you need to first run the traning code to obtain the parameters of the model, and then run the evaluation code:

|  Dataset  |                                            Video level                                           |                                         Shot level                                         |                                          Temporal point                                         |                                            Temporal                                           |                                      Temporal + spatial points                                     |                                           1 BB                                           |                                       Temp. + 1 BB                                       |                                         Temp. + 3 BBs                                         |                                            Fully supervised                                           |
|:---------:|:------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------:|
| UCF101-24 | [script](https://github.com/jalayrac/weakactionloc/blob/master/scripts/UCF101-24/video_level.sh) |                                              -                                             | [script](https://github.com/jalayrac/weakactionloc/blob/master/scripts/UCF101-24/temp_point.sh) | [script](https://github.com/jalayrac/weakactionloc/blob/master/scripts/UCF101-24/temporal.sh) | [script](https://github.com/jalayrac/weakactionloc/blob/master/scripts/UCF101-24/temp_sppoints.sh) | [script](https://github.com/jalayrac/weakactionloc/blob/master/scripts/UCF101-24/1BB.sh) | [script](https://github.com/jalayrac/weakactionloc/blob/master/scripts/UCF101-24/1BB.sh) | [script](https://github.com/jalayrac/weakactionloc/blob/master/scripts/UCF101-24/temp_3BB.sh) | [script](https://github.com/jalayrac/weakactionloc/blob/master/scripts/UCF101-24/fully_supervised.sh) |
|    DALY   |    [script](https://github.com/jalayrac/weakactionloc/blob/master/scripts/DALY/video_level.sh)   | [script](https://github.com/jalayrac/weakactionloc/blob/master/scripts/DALY/shot_level.sh) |                                                -                                                |    [script](https://github.com/jalayrac/weakactionloc/blob/master/scripts/DALY/temporal.sh)   |                                                  -                                                 |    [script](https://github.com/jalayrac/weakactionloc/blob/master/scripts/DALY/1BB.sh)   |    [script](https://github.com/jalayrac/weakactionloc/blob/master/scripts/DALY/1BB.sh)   |    [script](https://github.com/jalayrac/weakactionloc/blob/master/scripts/DALY/temp_3BB.sh)   |                                                   -                                                   |


NB: we provide the calibration files (threshold values) that were obtained by validation as described in the paper. We will release the code for this calibration in a future release.

