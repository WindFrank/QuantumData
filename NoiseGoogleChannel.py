import cirq
from BitAndPhaseFlipChannel import BitAndPhaseFlipChannel # 这里的调用，前面是文件名，后面才是类名

# Get a single-qubit bit-flip channel
bit_flip = cirq.bit_flip(p=0.1)
# 这个程序需要运行 cirq的21.1.2版本，网页中使用的pip install cirq --pre命令可能后面会失效，这里记录下
for i, kraus in enumerate(cirq.kraus(bit_flip)):
    print(f"Kraus operator {i + 1} is: \n", kraus, end="\n\n")

"""Example of using channels in a circuit."""
# See the number of qubits a channel acts on.
nqubits = bit_flip.num_qubits()
print(f"Bit flip channel acts on {nqubits} qubit(s).\n")

# Apply the channel to each qubit in a circuit.
circuit = cirq.Circuit(
    bit_flip.on_each(cirq.LineQubit.range(3))
)
print(circuit)

print('\n')
"""Example of controlling a channel."""
# Get the controlled channel.
controlled_bit_flip = bit_flip.controlled(num_controls=1)

# Use it in a circuit.
circuit = cirq.Circuit(
    controlled_bit_flip(*cirq.LineQubit.range(2))
)
print(circuit)

print('\n')

"""Get an asymmetric depolarizing channel."""
depo = cirq.asymmetric_depolarize(
    p_x=0.10,
    p_y=0.05,
    p_z=0.15,
)

circuit = cirq.Circuit(
    depo.on_each(cirq.LineQubit(0))
)
print(circuit)

print('\n')

"""Example of using the mixture protocol."""
# mixture protocol：混合协议，指的是p概率产生噪声，1-p概率不产生噪声
for prob, kraus in cirq.mixture(bit_flip):
    print(f"With probability {prob}, apply\n", kraus, end="\n\n")

"""The amplitude damping channel is an example of a channel without a mixture."""
# 部分信道没有混合协议，它们的克劳斯算符Kraus operator依赖于密度ρ的值。
# 下面是一个判断信道是否有混合协议的实例
channel = cirq.amplitude_damp(0.1)

if cirq.has_mixture(channel):
    print(f"Channel {channel} has a _mixture_ or _unitary_ method.")
else:
    print(f"Channel {channel} does not have a _mixture_ or _unitary_ method.")

print('\n')
"""Custom channels can be used like any other channels."""
# Custom channel自定义信道的使用示例：
bit_phase_flip = BitAndPhaseFlipChannel(p=0.05)

for prob, kraus in cirq.mixture(bit_phase_flip):
    print(f"With probability {prob}, apply\n", kraus, end="\n\n")

"""Example of using a custom channel in a circuit."""
circuit = cirq.Circuit(
    bit_phase_flip.on_each(*cirq.LineQubit.range(3))
)
print(circuit)

print('\n')

'''
cirq.DensityMatrixSimulator可以模拟任何噪声电路(即可以应用任何量子通道)  
因为它存储了完整的密度矩阵ρ。  
这种模拟策略通过直接应用每个量子信道的克劳斯算符来更新ρ态。  
'''

"""Simulating a circuit with the density matrix simulator."""
# Get a circuit.
# GridQubit.rect(i, j)：创建一个二维矩阵，行为i列为j，该二维矩阵中的每个位置都是一个|0>量子位；
# GridQubit(i, j)指返回i行j列位置的一个量子位，
# 在这里，相当于简单赋值|0>
qbit = cirq.GridQubit(0, 0)
circuit = cirq.Circuit(
    cirq.X(qbit),
    cirq.amplitude_damp(0.1).on(qbit)
)

# Display it.
print("Simulating circuit:")
print(circuit)

# Simulate with the density matrix simulator. 这个方法估计是为任何噪声电路找到其密度矩阵用的。
dsim = cirq.DensityMatrixSimulator()
rho = dsim.simulate(circuit).final_density_matrix

# Display the final density matrix.
print("\nFinal density matrix:")
print(rho)

print('\n')

'''
带有任意信道的噪声电路也可以用cirq.Simulator来模拟。 
当这样模拟一个信道时，一个Kraus算子被随机采样(根据概率分布)并应用于波函数。 
这种方法被称为“蒙特卡罗(波函数)模拟”或“量子轨迹”。

对于不支持cirq.mixture的信道，各克劳斯算子应用的概率取决于状态。 
相反，对于确实支持cirq.mixture的信道，各克劳斯算子应用的概率与状态无关。  
'''
"""Simulating a noisy circuit via Monte Carlo simulation."""
# Get a circuit.
qbit = cirq.NamedQubit("Q")
circuit = cirq.Circuit(cirq.bit_flip(p=0.5).on(qbit))

# Display it.
print("Simulating circuit:")
print(circuit)

# Simulate with the cirq.Simulator.
sim = cirq.Simulator()
psi = sim.simulate(circuit).dirac_notation()

# Display the final wavefunction.
print("\nFinal wavefunction:")
print(psi)

'''
如果想要知道输出是随机的，可以多次运行上面的程序。
由于在位翻转通道中p=0.5，您应该得到|0>大约一半的时间，|1>大约一半的时间。
也可以使用多次重复的run方法来查看此行为。  
'''

print('\n')

"""Example of Monte Carlo wavefunction simulation with the `run` method."""
circuit = cirq.Circuit(
    cirq.bit_flip(p=0.5).on(qbit),
    cirq.measure(qbit),
)
res = sim.run(circuit, repetitions=100)
print(res.histogram(key=qbit))

'''
通常电路是用酉运算定义的，但我们想用噪声来模拟它们。 在Cirq中插入噪声的方法有几种。  
对于任何电路，都可以调用with_noise方法在每一刻之后插入一个通道。  
'''
print('\n')
"""One method to insert noise in a circuit."""
# Define some noiseless circuit.
circuit = cirq.testing.random_circuit(
    qubits=3, n_moments=3, op_density=1, random_state=11
)

# Display the noiseless circuit.
print("Circuit without noise:")
print(circuit)

# Add noise to the circuit.
noisy = circuit.with_noise(cirq.depolarize(p=0.01))

# Display it.
print("\nCircuit with noise:")
print(noisy)

print('\n')

'''
with_noise方法从它的输入创建一个cirq.NoiseModel，并为每个时刻添加噪音。  
cirq.NoiseModel 可以被显式创建，并使用它向单个操作、单个时刻或一系列时刻添加噪声，如下所示。  
'''
"""Add noise to an operation, moment, or sequence of moments."""
# Create a noise model.
noise_model = cirq.NoiseModel.from_noise_model_like(cirq.depolarize(p=0.01))

# Get a qubit register.
qreg = cirq.LineQubit.range(2)

# Add noise to an operation.
op = cirq.CNOT(*qreg)
noisy_op = noise_model.noisy_operation(op)

# Add noise to a moment.
moment = cirq.Moment(cirq.H.on_each(qreg))
noisy_moment = noise_model.noisy_moment(moment, system_qubits=qreg)

# Add noise to a sequence of moments.
circuit = cirq.Circuit(cirq.H(qreg[0]), cirq.CNOT(*qreg))
noisy_circuit = noise_model.noisy_moments(circuit, system_qubits=qreg)

'''
每个“噪声方法”的输出是一个cirq.OP_TREE，
可以通过将其传递到cirq.Circuit构造器中来转换为电路。
例如，我们用下面的噪声矩创建一个电路。  
'''
"""Creating a circuit from a noisy cirq.OP_TREE."""
myCircuit = cirq.Circuit(noisy_moment)
print(myCircuit)

'''另一种技术是将噪声通道传递到密度矩阵模拟器，如下所示。  '''
"""Define a density matrix simulator with a noise model."""
noisy_dsim = cirq.DensityMatrixSimulator(
    noise=cirq.generalized_amplitude_damp(p=0.1, gamma=0.5)
)
'''
这不会显式地将通道添加到被模拟的电路中，但电路将被模拟，就像这些通道存在一样。  
 
除了这些通用方法外，信道可以像门一样随时添加到电路中。 通道可以是不同的，可以是相关的，可以作用于量子位元的子集，可以是自定义的，等等。
'''
"""Defining a circuit with multiple noisy channels."""
qreg = cirq.LineQubit.range(4)
circ = cirq.Circuit(
    cirq.H.on_each(qreg),
    cirq.depolarize(p=0.01).on_each(qreg),
    cirq.qft(*qreg),
    bit_phase_flip.on_each(qreg[1::2]),
    cirq.qft(*qreg, inverse=True),
    cirq.reset(qreg[1]),
    cirq.measure(*qreg),
    cirq.bit_flip(p=0.07).controlled(1).on(*qreg[2:]),
)

print("Circuit with multiple channels:\n")
print(circ)

'''
电路也可以用标准的方法进行修改，比如在电路的任何点添加通道。 
例如，为了模拟简单的状态准备错误，可以在电路的开始处添加位翻转通道，如下所示。  
'''
"""Example of inserting channels in circuits."""
circ.insert(0, cirq.bit_flip(p=0.1).on_each(qreg))
print(circ)
