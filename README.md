# causal-transformer
A causal transformer based LM.

Currently has the following features:
- coscine similarity based attention (i've found that this works better)
- gradient checkpointing (save that precious memory)
- intermediate losses (improves performance esp. in lower data scenarious) https://arxiv.org/pdf/2102.03216.pdf
- dynamic position bias (generalizes to much longer distances than anything else I've tried)
- multi-query attention (keys and values only have 1 head) https://arxiv.org/pdf/1911.02150.pdf
- token-Shift (feature dim is shifted forwards by 1) https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py and idea from https://github.com/BlinkDL/RWKV-LM
- talking-heads (noam shazeer paper, used in x-transformers, I've been just using this pre-softmax cus Ive found that's more stable, I added setting 'pre' | 'post' | 'both' for this)


Alot of this  comes from lucidrains/Phil Wang x-transformers library
