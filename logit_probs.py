import numpy as np
logits = [2.0, 1.0, 0.1]
exp_logits = np.exp(logits)
probs = exp_logits / np.sum(exp_logits)
print (probs)