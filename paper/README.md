
# Paper

This page documents the data from experiments in the paper.

```dos/```<br/>
Density of states of trained and shuffled networks. For each text file, the first column represents the energy $E$ and the second column denotes the density of states $ln g(E)/N$ normalized by the number of spins $N$.

```blocks/```<br/>
Density of states for trained and shuffled networks where Ising models are constructed using a varying number of transformer layers (e.g., `blocks4` means Ising models are constructed from four transformer layers).

```fraction/```<br/>
Density of states for networks where a different fraction of weights are shuffled (e.g., `fraction0.01` means $1$ percentage of values are shuffled).

```accuracy/```<br/>
Accuracy achieved during inference after shuffling weights of the trained networks. For each text file, the first column represents the shuffling fraction $f$ and the second column denotes the task metric (i.e., accuracy for ImageNet, and PPL for the language tasks).

```specificheat/```<br/>
The specific heat computed using the density of states in `dos/`. For each text file, the first column represents the temperature $T$ and the second column denotes the specific heat $C(T)$.
