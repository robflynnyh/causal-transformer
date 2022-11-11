# causal-transformer
A causal transformer based LM.

Currently has the following features:
- coscine similarity based attention (i've found that this works better)
- gradient checkpointing (save that precious memory)
- intermediate losses (improves performance esp. in lower data scenarious)
- stochastic depth (regularization for training extremely deep networks https://arxiv.org/pdf/2102.03216.pdf)
- dynamic position bias (generalizes to much longer distances than anything else I've tried)

credit to lucidrains/philwang as some of this (coscine similarity attention and dynamic pos) comes from his x-transformers library 
