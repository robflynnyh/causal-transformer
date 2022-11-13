# causal-transformer
A causal transformer based LM.

Currently has the following features:
- coscine similarity based attention (i've found that this works better)
- gradient checkpointing (save that precious memory)
- intermediate losses (improves performance esp. in lower data scenarious) https://arxiv.org/pdf/2102.03216.pdf
- dynamic position bias (generalizes to much longer distances than anything else I've tried)
- multi-query attention (keys and values only have 1 head) https://arxiv.org/pdf/1911.02150.pdf
- token-Shift (feature dim is shifted forwards by 1) https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py and idea from https://github.com/BlinkDL/RWKV-LM
- single-head MoS (mixture of softmaxes) Replaces multi-head attention with a single head and several learnt softmax temperatures

credit to lucidrains/philwang as some of this (coscine similarity attention and dynamic pos) comes from his x-transformers library. Here I use a learnt temperature for the coscine attention as described here https://arxiv.org/pdf/2010.04245.pdf rather than the grouped l2 norm in x-transformers. I've found initiating the temperature at 15.5 to work fine
