import numpy as np
import Transform as tr

a = np.array([5,5,3])
b = np.clip(a,0,3)
print(b)
print(b.flatten())