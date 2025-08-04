import numpy as np
logits = [2.0, 1.0, 0.1]
exp_logits = np.exp(logits)
probs = exp_logits / np.sum(exp_logits)
print (probs)

logits2 = [6.0, 2.0, 1.0, 0.05]
exp_logits2 = np.exp(logits2)
probs2 = exp_logits2 / np.sum(exp_logits2)
print(probs2)