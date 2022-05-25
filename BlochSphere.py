import numpy as np

#布洛赫球坐标转换
def sph2cart(theta, phi):
    # theta: polar angle
    # phi: azimuthal angle
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    return x,y,z

