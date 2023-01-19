# causal-transformer
A causal transformer based LM.

Currently has the following features:
- caching previous keys and values for - see test function caching_test() - allows for incremental inference or training like with transformer-xl 
- coscine similarity based attention (i've found that this works better) https://arxiv.org/abs/2010.04245
- gradient checkpointing (save that precious memory)
- intermediate losses (improves performance esp. in lower data scenarious) https://arxiv.org/pdf/2102.03216.pdf
- dynamic position bias (generalizes well to v long distances - works very similar to AliBi but is learnt so more flexible)
- multi-query attention (keys and values only have 1 head, increases utilization when paired with kv caching, used in PaLM model) https://arxiv.org/pdf/1911.02150.pdf
- token-Shift (feature dim is shifted forwards by 1) https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py and idea from https://github.com/BlinkDL/RWKV-LM faster convergence on character level LMing
- talking-heads (noam shazeer paper, used in x-transformers, I've been just using this pre-softmax cus Ive found that's more stable, I added setting 'pre' | 'post' | 'both' for this, I think paired with coscine similarity attention this makes the model less sensitive to its initial temperature hyperparam (having this at 15.5 has been fine for a range of sequence lengths)


Some of the code is taken or adapted from lucidrains/Phil Wangs x-transformers library
