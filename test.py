from infrastructure import *
import matplotlib.pyplot as plt
steps_b = 10
steps_r = 25
total_points = 1000

x_vals = np.linspace(0,2*np.pi,total_points)
D_t, D2 = transport_matrix(depth = 2*np.pi, total_points=total_points)
D = D_HJB(depth = 2*np.pi, total_points=total_points)
plt.plot(x_vals, -D_t @ np.cos(x_vals))
plt.plot(x_vals, np.sin(x_vals))
#plt.plot(x_vals, -D_t.T @ np.sin(x_vals))

plt.show()

print(-D_t.T)