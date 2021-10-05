from infrastructure import *
tot_points = 100
Mx = discrete_patches(10,tot_points)
inte = np.ones(tot_points).reshape(1, tot_points)
print(inte @ (Mx.M @ Mx.x))