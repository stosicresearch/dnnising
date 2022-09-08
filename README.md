# Ising models of deep neural networks 

## Introduction
This work (see arxiv paper) maps deep neural networks to classical Ising spin models, allowing them to be described using statistical thermodynamics. The density of states shows that structures emerge in the weights after they have been trained – well-trained networks span a much wider range of realizable energy values compared to poorly trained ones. Structures permeate throughout the entire network and cannot be observed in a few selected layers. Moreover, the energy values correlate to task accuracy, making it possible to distinguish models based on quality, something that is rather difficult without access to data. Lastly, thermodynamic properties such as specific heat are also studied.

## Data
Simulations are conducted on a range of pretrained deep neural networks available in [Huggingface](https://huggingface.co/), which cover both encoder- and decoder-based transformers of various sizes or language and vision tasks. The deep neural networks analyzed include: [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2), [OPT](https://huggingface.co/docs/transformers/model_doc/opt), [Bloom](https://huggingface.co/docs/transformers/model_doc/bloom), [BERT](https://huggingface.co/docs/transformers/model_doc/bert), [BeiT](https://huggingface.co/docs/transformers/model_doc/beit), [DeiT](https://huggingface.co/docs/transformers/model_doc/deit), and [ViT](https://huggingface.co/docs/transformers/model_doc/vit).

This repository provides data from the different experiments, which is organized as follows:
* `dos/` Density of states of trained and shuffled networks. For each text file, the first column represents the energy $E$ and the second column denotes the density of states $ln g(E)/N$ normalized by the number of spins $N$.
* `blocks/` Density of states for trained and shuffled networks where Ising models are constructed using a varying number of transformer layers (e.g., `blocks4` means Ising models are constructed from four transformer layers).
* `fraction/` Density of states for networks where a different fraction of weights are shuffled (e.g., `fraction0.01` means $1\%$ are shuffled).
* `accuracy/` Accuracy achieved during inference after shuffling weights of the trained networks. For each text file, the first column represents the shuffling fraction $f$ and the second column denotes the task metric (i.e., accuracy for ImageNet, and PPL for the language tasks).
* `specificheat/` The specific heat computed using the density of states in `dos/`. For each text file, the first column represents the temperature $T and the second column denotes the specific heat $C(T)$.

## Density of States
Below are shown the density of states for the various transformers both trained and after their values have been shuffled, representing untrained models without changing the distribution of weights. It can be observed that well-trained models span a much wider range of energies that can be realized compared to after shuffling, which represent poorly trained networks. To generate the plots, run `python dos.py` for the different models.

| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="1604" alt="" src="./plots/dos_opt-125m.png"> | <img width="1604" alt="" src="./plots/dos_opt-350m.png">|<img width="1604" alt="" src="./plots/dos_opt-1.3b.png">|
|<img width="1604" alt="" src="./plots/dos_gpt2.png"> | <img width="1604" alt="" src="./plots/dos_gpt2-medium.png">|<img width="1604" alt="" src="./plots/dos_bloom.png">|
|<img width="1604" alt="" src="./plots/dos_vit-base.png"> | <img width="1604" alt="" src="./plots/dos_deit-base.png">|<img width="1604" alt="" src="./plots/dos_beit-base.png">|
|<img width="1604" alt="" src="./plots/dos_bert-base.png"> | <img width="1604" alt="" src="./plots/dos_bert-large.png">| |

## Structures
Since the same distribution of values is guaranteed through shuffling, differences in the density of states must arise due to structure, or how weights are arranged in a neural network. The question then arises whether these structures exist in a subset of layers. Below are shown the difference in widths of the density of states, $\Delta = W_{train} - W_{shuffled}$ where $W=E_{max}-E_{min}$ , for different number of transformer layers $l$ that participate in the Ising model. From the plot, it can be observed that $W$ increases when adding more layers, which implies that structures appear throughout the entire network rather than in a few layers. To generate the plot, run `python blocks.py`.

<img width="1604" alt="" src="./plots/blocks.png">

## Shuffling and Accuracy
The density of states are compared between networks using different amounts of shuffling. Below are shown the widths $W$ of the density of states for various networks as a function of the fraction $f$ of values being shuffled. The plot shows that $W$ drops rapidly and approaches that of a random network ( $f=1$ ) after more than $20$ percent of the values have been shuffled. We also show percentage of task error achieved on tasks for the same networks, where it is found that $W$ decreases as the task error increases, which implies that the density of states correlates to accuracy on actual tasks. To generate the plots, run `python fraction.py` and `python accuracy.py`.

| | |
|:-------------------------:|:-------------------------:|
|<img width="1604" alt="" src="./plots/fraction.png"> | <img width="1604" alt="" src="./plots/accuracy.png">|

## Specific Heat
Lastly, the specific heat $C(T)$ is computed from the density of states for trained and shuffled networks, as shown below. The figure suggests that different networks acquire different specific heat curves, and the critical temperature $T_c$ (i.e., temperature $T$ that maximizes $C(T)$) also varies between networks, as well as between trained and shuffled. To generate the plots, run `python specificheat.py`.

| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="1604" alt="" src="./plots/C_opt-125m.png"> | <img width="1604" alt="" src="./plots/C_opt-350m.png">|<img width="1604" alt="" src="./plots/C_opt-1.3b.png">|
|<img width="1604" alt="" src="./plots/C_gpt2.png"> | <img width="1604" alt="" src="./plots/C_gpt2-medium.png">|<img width="1604" alt="" src="./plots/C_bloom.png">|
|<img width="1604" alt="" src="./plots/C_vit-base.png"> | <img width="1604" alt="" src="./plots/C_deit-base.png">|<img width="1604" alt="" src="./plots/C_beit-base.png">|
|<img width="1604" alt="" src="./plots/C_bert-base.png"> | <img width="1604" alt="" src="./plots/C_bert-large.png">| |


## Citation
If you use this work in academic research, we would appreciate citations to the following reference:

```
@misc{stosic2022ising,
title = {Ising models of deep neural networks}, 
author = {Dusan Stosic and Darko Stosic and Tatijana Stosic and Borko Stosic},
year = {2022},
}
```

## Contact
Dusan Stosic (dbstosic@gmail.com)
Darko Stosic (ddstosic@bu.edu)
Tatijana Stosic (tastosic@gmail.com)
Borko Stosic (borkostosic@gmail.com)
