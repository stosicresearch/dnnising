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

Allocate memory for storing the exchange coefficients `J` and spins `S`, where the are initialized to $S=1$.
```
void alloc(double**** J, int*** S, int** nodes, int num_layers, int num_nodes)
void py_alloc(int num_layers, int num_nodes)
```

Load the network weights from text files under `<dirname>/` and stores them into `J`
```
void read(double*** J, int* nodes, int num_layers, char* dirname)
void py_read()
```

Shuffle each `J` value at random `num_reps` times
```
void shuffle(double*** J, int* nodes, int num_layers, int num_reps)
void py_shuffle(int num_reps)
```

Compute energy of the system as $E=-\sum_{<l,i,j>} J_{lij} S_{lij}$.
```
double energy(double*** J, int** S, int* nodes, int num_layers)
py_energy()
```

Monte Carlo (quenching) step to minimize the energy
```
double mcmc_step(double e, double*** J, int** S, int* nodes, int num_layers)
double py_mcmc_step()
```

Save the current spin configuration `S` into text files under `spins/`
```
void save(int** S, int* nodes, int num_layers)
void py_save()
```

## Example
Below is an example in Python that loads a transformer model and runs Monte Carlo simulations to minimize its energy.

Import the `dnnising` library after compilation
```
import numpy as np
import dnnising
```

Define the parameters such as the diretory that holds the network weights, hidden size (number of neurons going ino the transformer block), and number of transformer layers
```
model = 'opt-125m'
hidden_size = 768
transformer_layers = 12
max_steps = 1000000;
num_shuffle = 10;
```

Allocate `J` and `S` (internal buffers to the library) and load the network weights into `J`. Note that we use `transformer_layers * 4`, since each transformer layer consists of 4 weight matrices
```
dnnising.py_alloc(transformer_layers * 4, hidden_size)
dnnising.py_read(model)
```

Shuffle `J` to get a random network with the same distribution of weights as the trained network
```
#dnnising.py_shuffle(num_shuffle)
```

Run Monte Carlo (quenching) simulations to minimize the energy of the system
```
e = 0.0
for i in range(max_steps):
    e = dnnising.py_mcmc_step(e)
print('Final energy:', e)
```

Store the spin configurations `S` as text files under `spins/`, which can take $S\pm1$ values.
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
