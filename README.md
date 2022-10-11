# transformer-from-scratch

My implementation of "Attention is All You Need" by Vaswani et al. [[1]](#1) using the JAX Framework.

Please see model code [here](https://github.com/FynnSu/transformer-from-scratch/blob/main/src/model.py).


My model code defines a number of generator functions that take as input a config dictionary and return two functions:
1. Function to compute (i.e. feedforward network, dropout layer, encoder, etc)
2. gen_params() function to generate initial weights for the compute function




## References
<a id="1">[1]</a> 
Vaswani et al. 2017
[Attention Is All You Need.](https://arxiv.org/abs/1706.03762)
arXiv.
