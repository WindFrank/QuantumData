from qiskit.visualization import plot_bloch_vector
from BlochSphere import sph2cart

from matplotlib import pyplot as plt

# 布洛赫球使用
theta = 0 # 0 np.pi np.pi/2
phi = 0 # 0 0 np.pi
x,y,z = sph2cart(theta, phi)
plot_bloch_vector([x, y, z])
plt.show()
