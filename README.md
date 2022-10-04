# Ising models of deep neural networks 

## Introduction
While deep neural networks have emerged as a disruptive technology capable of solving a range of complex problems, the underlying mechanisms that makes them so effective continue to elude us. For example, there remains no clear way to distinguish a well- from a poorly- trained network other than through evaluation on tasks, which by itself can be insufficient to determine whether a network has been properly trained.

Deep neural networks are by construction reminiscent of magnetic model systems where nodes are connected by couplings (weights), giving rise to collective behavior that cannot be described by their individual parts. [This](https://arxiv.org/abs/2209.08678) work borrows the concepts of statistical physics (and thermodynamics) to analyze deep neural networks by mapping them to a well-known problem in statistical physics, the Ising model. With this formulation, the weights of a neural network are taken to represent exchange interactions between spins represented by the nodes of the network, and the system can be studied using various properties of spin glass models.

This repository releases code for mapping transformer networks into Ising models, as well as running Monte Carlo simulations to minimize obtain the minimum energy spin configurations. The density of states from Wang-Landau simulations are also made public.

## Network weights
The network weights can be downloaded from [Huggingface](https://huggingface.co/), which include [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2).

An example for loading weights from one of the [OPT](https://huggingface.co/docs/transformers/model_doc/opt) models is provided in `convert_opt.py`, which can be used as the basis for loading other weights from other networks (with slight changes to the layer names, etc).

## Library
The library is contained in `dnnising.h`, along with examples of integration in C and Python. For C, compile and run the Microsoft Visual Studio project. For Python,
compile the wrapper library using ```python setup build && cp build/lib*/dnnising* .``` and run the example ```python dnnising.py```.

The library exposes a range of functions:

Allocates (internally) memory for storing the exchange coefficients, ```J```, and spins, ```S```.
```
void alloc(double**** J, int*** S, int** nodes, int num_layers, int num_nodes)
void py_alloc(int num_layers, int num_nodes)
```
<br/>

Reads the network weights from text files under ```dirname`` and stores them into ```J```.
```
void read(double*** J, int* nodes, int num_layers, char* dirname)
void py_read()
```
<br/>

Shuffles each ```J``` value ```num_reps``` times.
```
void shuffle(double*** J, int* nodes, int num_layers, int num_reps)
void py_shuffle(int num_reps)
```
<br/>

A Monte Carlo (quenching) step to minimize the energy.
```
double mcmc_step(double e, double*** J, int** S, int* nodes, int num_layers)
double py_mcmc_step()
```

Saves the current spin configuration `S` into text files under ```spins/```.
```
void save(int** S, int* nodes, int num_layers)
void py_save()
```

## Example
Below we go over a Python example that loads a 125M GPT transformer model and runs Monte Carlo simulations to minimize its energy.

First, we need to import the library.
```
import numpy as np
import dnnising
```

Next we define the parameters, such as the model file, hidden size (i.e., number of neurons going ino the transformer block), number of transformer layers, etc.
```
model = 'opt-125m'
hidden_size = 768
transformer_layers = 12
max_steps = 1000000;
num_shuffle = 10;
```

Then we allocate the internal buffers (`J` and `S`) and load the network weights into J. Note we pass in `transformer_layers * 4`, since a transformer layer consists of four unique weight matrices.
```
dnnising.py_alloc(transformer_layers * 4, hidden_size)
dnnising.py_read(model)
```

We can then optionally shuffle J when trying to compare between trained and random networks.
```
#dnnising.py_shuffle(num_shuffle)
```

We use Monte Carlo (quenching) simulations to minimize the energy fo the system across `max_steps` sweeps.
```
e = 0.0
for i in range(max_steps):
    e = dnnising.py_mcmc_step(e)
print('Final energy:', e)
```

Lastly, the spin configuration `S` can be stored under `spins/`.
```
dnnising.py_save()
```

## Data
The data and results from experiments used in the paper are organized under `paper/`.

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
