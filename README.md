# Ising models of deep neural networks 

## Introduction
This [work](https://arxiv.org/abs/2207.01169) maps deep neural networks to classical Ising spin models, allowing them to be described using statistical thermodynamics. The density of states shows that structures emerge in the weights after they have been trained – well-trained networks span a much wider range of realizable energy values compared to poorly trained ones. Structures permeate throughout the entire network and cannot be observed in a few selected layers. Moreover, the energy values correlate to task accuracy, making it possible to distinguish models based on quality, something that is rather difficult without access to data. Lastly, thermodynamic properties such as specific heat are also studied.

## Results
Simulations are conducted on a range of pretrained deep neural networks available in [Huggingface](https://huggingface.co/), which cover both encoder- and decoder-based transformers of various sizes or language and vision tasks. The deep neural networks analyzed include: [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2), [OPT](https://huggingface.co/docs/transformers/model_doc/opt), [Bloom](https://huggingface.co/docs/transformers/model_doc/bloom), [BERT](https://huggingface.co/docs/transformers/model_doc/bert), [BeiT](https://huggingface.co/docs/transformers/model_doc/beit), [DeiT](https://huggingface.co/docs/transformers/model_doc/deit), and [ViT](https://huggingface.co/docs/transformers/model_doc/vit).

### Density of States
Below are shown the density of states for the various transformers both trained and after their values have been shuffled, representing untrained models without changing the distribution of weights. The raw results can be found under ```dos/```, where the first column represents the energy $E/N$ and the second column the density of states, $ln(E)/N$, normalized by the number of spins $N$.

| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="1604" alt="" src="./plots/dos_opt-125m.png"> | <img width="1604" alt="" src="./plots/dos_opt-350m.png">|<img width="1604" alt="" src="./plots/dos_opt-1.3b.png">|
|<img width="1604" alt="" src="./plots/dos_gpt2.png"> | <img width="1604" alt="" src="./plots/dos_gpt2-medium.png">|<img width="1604" alt="" src="./plots/dos_bloom.png">|
|<img width="1604" alt="" src="./plots/dos_vit.png"> | <img width="1604" alt="" src="./plots/dos_deit.png">|<img width="1604" alt="" src="./plots/dos_beit.png">|
|<img width="1604" alt="" src="./plots/dos_bert-base.png"> | <img width="1604" alt="" src="./plots/dos_bert-large.png">| |


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
