from infrastructure import *
import numpy as np
import matplotlib.pyplot as plt
Mx = discrete_patches(10,20)



D = fin_diff_mat_periodic(99)

x_vals = np.linspace(0,10,20)
print(np.sum(Mx.M @ x_vals))
#plt.show()


