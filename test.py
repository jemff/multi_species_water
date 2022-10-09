from infrastructure import *
import matplotlib.pyplot as plt
steps_b = 10
steps_r = 25
total_points = 100

x_vals = np.linspace(0, 24*60,total_points+1)[0:-1]
D_t, D2 = transport_matrix(depth = 2*np.pi, total_points=total_points)
#D = D_HJB(depth = 2*np.pi, total_points=total_points)
D = spectral_periodic(total_points, length = 24*60)
plt.plot(x_vals, -D @ np.cos(2*np.pi*x_vals/(24*60)))
print(D)
#plt.plot(x_vals, np.sin(2*np.pi*x_vals/(24*60)))
#plt.plot(x_vals, -D_t.T @ np.sin(x_vals))

#plt.show()
print(np.linspace(1,21,11))
#print(-D_t.T)