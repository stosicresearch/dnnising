# Ising models of deep neural networks 

## Introduction
While deep neural networks have emerged as a disruptive technology capable of solving a range of complex problems, the underlying mechanisms that makes them so effective continue to elude us. For example, there remains no clear way to distinguish a well- from a poorly- trained network other than through evaluation on tasks, which by itself can be insufficient to determine whether a network has been properly trained.

Deep neural networks are by construction reminiscent of magnetic model systems where nodes are connected by couplings (weights), giving rise to collective behavior that cannot be described by their individual parts. This work (see arXiv paper) borrows the concepts of statistical physics (and thermodynamics) to analyze deep neural networks by mapping them to a well-known problem in statistical physics, the Ising model. With this formulation, the weights of a neural network are taken to represent exchange interactions between spins represented by the nodes of the network, and the system can be studied using various properties of spin glass models.

## Setup
We release code for mapping transformer networks into Ising models in the `dnnising.h` header with examples of integration in C and PyTorch. Below are instructions for running examples in the python environment.

1. Download and extract the network weights
```
python extract_gpt.py
```

1. Compile the wrapper library.
```
python setup build && cp build/lib*/dnnising* .
```

2. Run the example
```
python dnnising.py
```

## Library
The library exposes a range of functions for mapping transformer networks to Ising models:

* ```py_alloc```: Allocates (internally) memory for storing the exchange coefficients, J, and spins, S.
* ```py_read```: Reads the network weights from a file and stores it into J.
* ```py_shuffle```: Shuffles the values in J.
* ```py_mcmc_step```: Runs a Monte Carlo (quenching) step that minimizes the energy.
* ```py_save```: Saves the current spin configuration S into files.

```
import numpy as np
import dnnising

# Parameters.
model = 'opt-125m'
hidden_size = 768
transformer_layers = 12
max_steps = 1000000;
num_shuffle = 10;

# Allocate and read model.
dnnising.py_alloc(transformer_layers * 4, hidden_size)
dnnising.py_read(model)

# Shuffle weights.
#dnnising.py_shuffle(num_shuffle)

# Minimize energy.
e = 0.0
for i in range(max_steps):
    e = dnnising.py_mcmc_step(e)
print('Final energy:', e)

# Save spins.
dnnising.py_save()
```

## Data
Simulations are conducted on a range of pretrained deep neural networks available in [Huggingface](https://huggingface.co/), which cover both encoder- and decoder-based transformers of various sizes or language and vision tasks. The deep neural networks analyzed include: [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2), [OPT](https://huggingface.co/docs/transformers/model_doc/opt), [Bloom](https://huggingface.co/docs/transformers/model_doc/bloom), [BERT](https://huggingface.co/docs/transformers/model_doc/bert), [BeiT](https://huggingface.co/docs/transformers/model_doc/beit), [DeiT](https://huggingface.co/docs/transformers/model_doc/deit), and [ViT](https://huggingface.co/docs/transformers/model_doc/vit).

This repository provides data from the different experiments, which is organized as follows:
* `dos/` Density of states of trained and shuffled networks. For each text file, the first column represents the energy $E$ and the second column denotes the density of states $ln g(E)/N$ normalized by the number of spins $N$.
* `blocks/` Density of states for trained and shuffled networks where Ising models are constructed using a varying number of transformer layers (e.g., `blocks4` means Ising models are constructed from four transformer layers).
* `fraction/` Density of states for networks where a different fraction of weights are shuffled (e.g., `fraction0.01` means $1$ percentage of values are shuffled).
* `accuracy/` Accuracy achieved during inference after shuffling weights of the trained networks. For each text file, the first column represents the shuffling fraction $f$ and the second column denotes the task metric (i.e., accuracy for ImageNet, and PPL for the language tasks).
* `specificheat/` The specific heat computed using the density of states in `dos/`. For each text file, the first column represents the temperature $T$ and the second column denotes the specific heat $C(T)$.

## Citation
If you use this work in academic research, we would appreciate citations to the following reference:

```
@article{stosic2022ising,
title = {Ising models of deep neural networks}, 
author = {Dusan Stosic and Darko Stosic and Tatijana Stosic and Borko Stosic},
year = {2022},
}
```

## Contact
Dusan Stosic (dbstosic@gmail.com)<br/>
Darko Stosic (ddstosic@bu.edu)<br/>
Borko Stosic (borkostosic@gmail.com)
