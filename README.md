

# 1. Official Code
Official PyTorch implementation of AROMA | [Accepted at Neurips 2024](https://openreview.net/forum?id=Aj8RKCGwjE&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2024%2FConference%2FAuthors%23your-submissions))

<a href="https://arxiv.org/abs/2406.02176"><img
src="https://img.shields.io/badge/arXiv-AROMA-b31b1b.svg" height=25em></a>

<p float="center">
  <img src="./assets/new_aroma_inference_v2.jpg" width="800"/>
</p>

To cite our work:

```
@inproceedings{
serrano2025zebra,
title={Zebra: In-Context Generative Pretraining for Solving Parametric {PDE}s},
author={Louis Serrano and Armand Kassa{\"\i} Koupa{\"\i} and Thomas X Wang and Pierre ERBACHER and Patrick Gallinari},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=22kNOkkokU}
}
```

# 2. Code installation and setup
## zebra installation
```
conda create -n zebra python=3.9.0
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

# 3. Data

We will shortly push the datasets used in this paper on HuggingFace (https://huggingface.co/sogeeking) and provide scripts to **download** them directly from there in the folder `download_dataset`.


# 4. Run experiments 

The code runs only on GPU. We provide sbatch configuration files to run the training scripts. They are located in `bash` and are organized by datasets.
We expect the user to have wandb installed in its environment for monitoring. 
In Zebra, the first step is to launch an tokenizer.py training, in order to learn a finite vocabulary of physical phenomena. The weights of the tokenizer model are automatically saved under its `run_name`.
For the second step, i.e. for training the language model with an in-context pretraining, we need to use the previous `run_name` as input to the config file to load the tokenizer model. The `run_name` can be set in the config file, but can also be generated randomly by default with wandb.
We provide examples of the python scripts that need to be run in each bash folder.

For instance, for `advection` we need to first train the VQVAE:
`sbatch bash/burgers/tokenizer.sh`
and then once we specified the correct run_name in the config:
`sbatch bash/burgers/llama.sh`

# Acknowledgements

This project would not have been possible without these awesome repositories:
* Transformer implementation from hugging face: https://github.com/huggingface/transformers
* MAGVIT implementation from lucidrains: https://github.com/lucidrains/magvit2-pytorch
* PDE Arena : https://github.com/pdearena/pdearena











