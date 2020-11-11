from simple_quad_MODEL import simple_quad_model as sq
import numpy as np

I = {'Ix': 8.276e-3, 'Iy': 8.276e-3, 'Iz': 1.612e-2}
quad_model = sq(0.9, 10, I, 0.175)
F = np.zeros(4)
for i in range(50):
    quad_model.step(F)
