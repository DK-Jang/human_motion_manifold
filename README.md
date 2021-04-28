# Constructing Human Motion Manifold with Sequential Networks in PyTorch (Official)
![Python](https://img.shields.io/badge/Python->=3.6-Blue?logo=python)  ![Pytorch](https://img.shields.io/badge/PyTorch->=1.5.0-Red?logo=pytorch)

<p align="center"><img src="images/random_sampling.gif" align="center"> <br></p>

> **Constructing Human Motion Manifold with Sequential Networks**<br>
> [Deok-Kyeong Jang](https://github.com/DK-Jang), [Sung-Hee Lee](http://motionlab.kaist.ac.kr/?page_id=41)<br>
> Eurographics 2021 Conference, in Computer Graphics Forum 2020<br>

> Paper: https://arxiv.org/abs/2005.14370<br>
> Project: http://motionlab.kaist.ac.kr/?page_id=5962<br>
> Video: https://www.youtube.com/watch?v=DPXnidbmtvs<br>

> **Abstract:** This paper presents a novel recurrent neural network-based method to construct a latent motion manifold that can represent a wide range of human motions in a long sequence. We introduce several new components to increase the spatial and temporal coverage in motion space while retaining the details of motion capture data. These include new regularization terms for the motion manifold, combination of two complementary decoders for predicting joint rotations and joint velocities, and the addition of the forward kinematics layer to consider both joint rotation and position errors. In addition, we propose a set of loss terms that improve the overall quality of the motion manifold from various aspects, such as the capability of reconstructing not only the motion but also the latent  manifold vector, and the naturalness of the motion through adversarial loss.
These components contribute to creating compact and versatile motion manifold that allows for creating new motions by performing random sampling and algebraic operations, such as interpolation and analogy, in the latent motion manifold.

## Requirements
- Pytorch >= 1.5
- Tensorboard 2.4.1
- h5py 2.9.0
- tqdm 4.35.0
- PyYAML 5.1.2
- matplotlib 3.3.4

## Installation
Clone this repository and create environment:

```bash
git clone https://github.com/DK-Jang/human_motion_manifold.git
cd human_motion_manifold
conda create -n motion_manifold python=3.6
conda activate motion_manifold
```
Install PyTorch >= 1.5 and torchvision from [PyTorch](https://pytorch.org/).
Install the other dependencies:
```bash
pip install -r requirements.txt 
```

## Datasets and pre-trained networks
To train Human Motion Manifold network, please download the dataset.
To run the demo, please download the dataset and pre-trained weight both.

<b>H3.6M dataset.</b> To download the [H3.6M](https://drive.google.com/file/d/1HNcgnCMOZ9p6WR-lsKhLOQhHbgOjZHhg/view?usp=sharing) dataset(npz) from Google Drive. Then place the npz file directory within `dataset/`. 
After that, run the following commands:

```bash
cd dataset
python make_train_test_folder.py
```

<b>Pre-trained weight.</b> To download the [weight](https://drive.google.com/file/d/1M05ECR04iB-NTjElF7xin8nm8-_Xq1Iw/view?usp=sharing) from Google Drive. Then place the pt file directory within `pretrained/pth/`.

## How to run the demo
After downloading the pre-trained weights, you can run the demo.
- <b>Reconsturction</b> motion, run following commands:
```bash
python reconstruction.py --config pretrained/info/config.yaml   # generate motions
python result2bvh.py --bvh_dir ./pretrained/output/recon/bvh \
                     --hdf5_path ./pretrained/output/recon/m_recon.hdf5    # hdf5 to bvh 
```
Generated motions(hdf5 format) will be placed under `./pretrained/output/recon/*.hdf5`. \
`m_gt.hdf5`: ground-truth motion, \
`m_recon.hdf5`: generated from joint rotation decoder, \
`m_recon_vel.hdf5`: generated from joint velocity decoder. \
Generated motions(bvh format) from joint rotation decoder will be placed under `./pretrained/output/recon/bvh/batch_*.bvh`.

- <b>Random sample</b> motions from motion manifold:
```bash
python random_sample.py --config pretrained/info/config.yaml
python result2bvh.py --bvh_dir ./pretrained/output/random_sample/bvh \
                     --hdf5_path ./pretrained/output/random_sample/m_recon.hdf5
```
Generated motions will be placed under `./pretrained/output/random_sample/*.hdf5`. \
Generated motions(bvh format) from joint rotation decoder will be placed under `./pretrained/output/random_sampling/bvh/batch_*.bvh`.


### Todo
- Denosing
- Interpolation
- Analogy

<!-- - <b>Denosing</b> motion data by projecting it to the latent motion manifold:
```bash
python remove_noise.py --config pretrained/info/config.yaml
```
Generated motions will be placed under `./pretrained/output/recon/*.hdf5`. -->

## How to train
To train human motion manifold networks from the scratch, run the following commands.
```bash
python train.py --config configs/H3.6M.yaml
```
Trained networks will be placed under `./motion_manifold_network/`

## Visualization
Easy way to visualize reconstruction results using matplotlib. You should save demo results as world positions.
For example:
```bash
python reconstruction.py --config pretrained/info/config.yaml \
                         --output_representation positions_world
python visualization.py --viz_path ./pretrained/output/recon/m_recon.hdf5
```

## Citation
If you find this work useful for your research, please cite our paper:

```
@inproceedings{jang2020constructing,
  title={Constructing human motion manifold with sequential networks},
  author={Jang, Deok-Kyeong and Lee, Sung-Hee},
  booktitle={Computer Graphics Forum},
  volume={39},
  number={6},
  pages={314--324},
  year={2020},
  organization={Wiley Online Library}
}
```

## Acknowledgements
This repository contains pieces of code from the following repositories: \
[QuaterNet: A Quaternion-based Recurrent Model for Human Motion](https://github.com/facebookresearch/QuaterNet). \
[A Deep Learning Framework For Character Motion Synthesis and Editing](http://theorangeduck.com/page/deep-learning-framework-character-motion-synthesis-and-editing).