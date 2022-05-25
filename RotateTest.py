import numpy as np

# rotate pi/2 along x_hat
angle = np.pi/2
I = np.array([[1,0],[0,1]])
sigma_x = np.array([[0,1],[1,0]])
rotation = np.cos(angle/2)*I - 1j*np.sin(angle/2)*sigma_x

initial_state = np.array([[1],[0]]) #|0> state
final_state = rotation @ initial_state # 应该是量子里使用的乘积运算了，这里让它与状态相乘完成旋转
print(final_state)