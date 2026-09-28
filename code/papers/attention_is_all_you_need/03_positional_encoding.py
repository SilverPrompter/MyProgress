
import numpy as np


# PE(pos, 2i) = sin( pos / 10000^(2i /d_model))
# PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model))

seq_len = 3
d_model = 4

PE = np.zeros((seq_len, d_model))

X = np.array([
    [1.0,0.0,0.0,0.0],
    [0.0,1.0,0.5,0.0],
    [0.0,0.5,1.0,0.0],
])

for pos in range(seq_len):
    for i in range(0, d_model,2):
        angle = pos / (10000 ** ( i / d_model))
        PE[pos , i] = np.sin(angle)
        PE[pos, i+1] = np.cos(angle)
print(PE)

X_with_position = X + PE
print(X_with_position)
