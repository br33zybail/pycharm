import numpy as np
y_true = [0, 1, 0]
y_pred = [0.07, 0.33, 0.6]
loss = -sum(y_t * np.log(y_p) for y_t, y_p in zip(y_true, y_pred))
print("Loss:\n", loss)