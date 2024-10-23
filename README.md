

# 0. Official Code
Official PyTorch implementation of AROMA | [Accepted at Neurips 2024](https://openreview.net/forum?id=Aj8RKCGwjE&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2024%2FConference%2FAuthors%23your-submissions))

<a href="https://arxiv.org/abs/2406.02176"><img
src="https://img.shields.io/badge/arXiv-AROMA-b31b1b.svg" height=25em></a>

<p float="center">
  <img src="./assets/new_aroma_inference_v2.jpg" width="800"/>
</p>

To cite our work:

```
@article{serrano2024aroma,
  title={AROMA: Preserving Spatial Structure for Latent PDE Modeling with Local Neural Fields},
  author={Serrano, Louis and Wang, Thomas X and Naour, Etienne Le and Vittaut, Jean-No{\"e}l and Gallinari, Patrick},
  journal={38th Conference on Neural Information Processing Systems (NeurIPS 2024)},
  year={2024}
}
```

# 1. Code installation and setup
## aroma installation
```
conda create -n aroma python=3.9.0
pip install -e .
```

## setup wandb config example

add to your `~/.bashrc`
```
export WANDB_API_TOKEN=your_key
export WANDB_DIR=your_dir
export WANDB_CACHE_DIR=your_cache_dir
export MINICONDA_PATH=your_anaconda_path
```

# 2. Data

We detail the **sources** of the datasets used during in this paper:

* CylinderFlow and AirfoilFlow datasets can be downloaded from : https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
* Navier Stokes 1e-3 and Shallow Water data can be generated using https://github.com/mkirchmeyer/DINo 
* Navier-Stokes 1e-4 and Navier-Stokes 1e-5 can be downloaded from https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-
* Burgers dataset can be generated from https://github.com/brandstetter-johannes/MP-Neural-PDE-Solvers

We uploaded most of the datasets on Hugging Face (https://huggingface.co/sogeeking) and provide scripts to **download** them directly from there in the folder `download_dataset`.
Therefore you can use those scripts to download efficiently the data.


# 3. Run experiments 

The code runs only on GPU. We provide sbatch configuration files to run the training scripts. They are located in `bash` and are organized by datasets.
We expect the user to have wandb installed in its environment to ease the 2-step training. 
For all tasks, the first step is to launch an inr.py training. The weights of the inr model are automatically saved under its `run_name`.
For the second step, i.e. for training the dynamics or inference model, we need to use the previous `run_name` as input to the config file to load the inr model. The `run_name` can be set in the config file, but is generated randomly by default with wandb.
We provide examples of the python scripts that need to be run in each bash folder.

For instance, for `burgers` we need to first train the VAE:
`sbatch bash/burgers/inr_burgers.sh`
and then once we specified the correct run_name in the config:
`sbatch bash/burgers/refiner_burgers.sh`

# Acknowledgements

This project would not have been possible without these awesome repositories:
* Perceiver implementation from lucid rains: https://github.com/lucidrains/perceiver-pytorch
* PDE refiner : https://github.com/pdearena/pdearena
* DIT implementation from Facebook : https://github.com/facebookresearch/DiT











