# Ising models of deep neural networks 

## Introduction
While deep neural networks have emerged as a disruptive technology capable of solving a range of complex problems, the underlying mechanisms that makes them so effective continue to elude us. For example, there remains no clear way to distinguish a well- from a poorly- trained network other than through evaluation on tasks, which by itself can be insufficient to determine whether a network has been properly trained.

Deep neural networks are by construction reminiscent of magnetic model systems where nodes are connected by couplings (weights), giving rise to collective behavior that cannot be described by their individual parts. [This](https://arxiv.org/abs/2209.08678) work borrows the concepts of statistical physics (and thermodynamics) to analyze deep neural networks by mapping them to a well-known problem in statistical physics, the Ising model. With this formulation, the weights of a neural network are taken to represent exchange interactions between spins represented by the nodes of the network, and the system can be studied using various properties of spin glass models.

This repository releases code for mapping transformer networks into Ising models, as well as running Monte Carlo simulations to minimize obtain the minimum energy spin configurations. The density of states from Wang-Landau simulations are also made public.

## Network weights
First, the network weights need to be downloaded from [Huggingface](https://huggingface.co/), which include [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2), [OPT](https://huggingface.co/docs/transformers/model_doc/opt), [Bloom](https://huggingface.co/docs/transformers/model_doc/bloom), [BERT](https://huggingface.co/docs/transformers/model_doc/bert), [BeiT](https://huggingface.co/docs/transformers/model_doc/beit), [DeiT](https://huggingface.co/docs/transformers/model_doc/deit), and [ViT](https://huggingface.co/docs/transformers/model_doc/vit).

An example for loading weights from ```OPT``` is provided as follows:
```
python convert_opt.py
```

## Library
The library is contained in `dnnising.h`, along with examples of integration in C and Python. Below are instructions for running the examples in Python.

For C,
Compile and run the Microsoft Visual Studio project.

For Python,
Compile the wrapper library using ```python setup build && cp build/lib*/dnnising* .``` and run the example ```python dnnising.py```

The library exposes a range of functions:

* ```py_alloc```: Allocates (internally) memory for storing the exchange coefficients, J, and spins, S.
* ```py_read```: Reads the network weights from a file and stores it into J.
* ```py_shuffle```: Shuffles the values in J.
* ```py_mcmc_step```: Runs a Monte Carlo (quenching) step that minimizes the energy.
* ```py_save```: Saves the current spin configuration S into files.


Below is a Python example that loads a 125M GPT transformer network and minimizes runs Monte Carlo simulations to find the spin configuration that minimizes the energy.
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
The data from experiments used in the paper are organized as follows:
* `paper/dos/` Density of states of trained and shuffled networks. For each text file, the first column represents the energy $E$ and the second column denotes the density of states $ln g(E)/N$ normalized by the number of spins $N$.
* `paper/blocks/` Density of states for trained and shuffled networks where Ising models are constructed using a varying number of transformer layers (e.g., `blocks4` means Ising models are constructed from four transformer layers).
* `paper/fraction/` Density of states for networks where a different fraction of weights are shuffled (e.g., `fraction0.01` means $1$ percentage of values are shuffled).
* `paper/accuracy/` Accuracy achieved during inference after shuffling weights of the trained networks. For each text file, the first column represents the shuffling fraction $f$ and the second column denotes the task metric (i.e., accuracy for ImageNet, and PPL for the language tasks).
* `paper/specificheat/` The specific heat computed using the density of states in `dos/`. For each text file, the first column represents the temperature $T$ and the second column denotes the specific heat $C(T)$.

## Citation
If you use this work in academic research, we would appreciate citations to the following reference:

```
@misc{stosic2022ising,
title = {Ising models of deep neural networks}, 
author = {Dusan Stosic and Darko Stosic and Borko Stosic},
year = {2022},
eprint = {2209.08678},
archivePrefix = {arXiv},
primaryClass = {cond-mat.stat-mech}
}
```

## Contact
Dusan Stosic (dbstosic@gmail.com)<br/>
Darko Stosic (ddstosic@bu.edu)<br/>
Borko Stosic (borkostosic@gmail.com)
