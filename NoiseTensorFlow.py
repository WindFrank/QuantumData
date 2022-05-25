import random
import cirq
import sympy
import tensorflow_quantum as tfq
import tensorflow as tf
import numpy as np
# Plotting
import matplotlib.pyplot as plt
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
'''
1.理解量子噪声  
1.1基本电路噪声  
量子计算机上的噪声会影响你能够从中测量的位串样本。 一种直观的方式是，嘈杂的量子计算机会在随机的地方“插入”、“删除”或“替换”门，
不幸的是，在实践中，几乎不可能知道电路可能出错的所有方式和它们的确切概率。
你可以做一个简化的假设，在电路中的每个操作之后，都有某种通道粗略地捕捉操作可能出错的情况。
你可以快速创建一个带有噪声的电路:
'''

def x_circuit(qubits):
  """Produces an X wall circuit on `qubits`."""
  return cirq.Circuit(cirq.X.on_each(*qubits))

def make_noisy(circuit, p):
  """Add a depolarization channel to all qubits in `circuit` before measurement."""
  return circuit + cirq.Circuit(cirq.depolarize(p).on_each(*circuit.all_qubits()))

my_qubits = cirq.GridQubit.rect(1, 2)
my_circuit = x_circuit(my_qubits)
my_noisy_circuit = make_noisy(my_circuit, 0.5)
print(my_circuit)
print(my_noisy_circuit)
print('\n')

'''
You can examine the noiseless density matrix ρ with:
你可以检验无噪声密度矩阵ρ  
'''
rho = cirq.final_density_matrix(my_circuit)
np.round(rho, 3)

'''
以及有噪声密度矩阵ρ
'''
rho = cirq.final_density_matrix(my_noisy_circuit)
np.round(rho, 3)

'''
Comparing the two different ρ's you can see that the noise has impacted the amplitudes of the state 
(and consequently sampling probabilities).  
In the noiseless case you would always expect to sample the |11> state. 
But in the noisy state there is now a nonzero probability of sampling |00>or|01>or|10>as well:
比较两个不同的ρ值，你可以看到噪声影响了状态的振幅(并因此影响了抽样概率)。 
在无噪声的情况下，你总是期望采样|11>状态。
但在噪声状态下，|00>或|01>或|10>的采样概率是非零的: 
'''

"""Sample from my_noisy_circuit."""
def plot_samples(circuit):
  samples = cirq.sample(circuit + cirq.measure(*circuit.all_qubits(), key='bits'), repetitions=1000)
  freqs, _ = np.histogram(samples.data['bits'], bins=[i+0.01 for i in range(-1,2** len(my_qubits))])
  plt.figure(figsize=(10,5))
  plt.title('Noisy Circuit Sampling')
  plt.xlabel('Bitstring')
  plt.ylabel('Frequency')
  plt.bar([i for i in range(2** len(my_qubits))], freqs, tick_label=['00','01','10','11'])

plot_samples(my_noisy_circuit)
plt.show()

'''
Without any noise you will always get |11>:
若没有噪声，您总会获得|11>：
'''
"""Sample from my_circuit."""
plot_samples(my_circuit)
plt.show()

'''
If you increase the noise a little further 
it will become harder and harder to distinguish the desired behavior (sampling |11> ) from the noise:
如果你进一步增加噪音，它将变得越来越难从噪音中区分想要的行为(采样|11>):  
'''
my_really_noisy_circuit = make_noisy(my_circuit, 0.75)
plot_samples(my_really_noisy_circuit)
plt.show()

'''
2. Basic noise in TFQ
With this understanding of how noise can impact circuit execution, 
you can explore how noise works in TFQ. 
TensorFlow Quantum uses monte-carlo / trajectory based simulation as an alternative to density matrix simulation. 
This is because the memory complexity of density matrix simulation 
limits large simulations to being <= 20 qubits with traditional full density matrix simulation methods. 
Monte-carlo / trajectory trades this cost in memory for additional cost in time.
The backend='noisy' option available 
 to all tfq.layers.Sample, tfq.layers.SampledExpectation and tfq.layers.Expectation 
 (In the case of Expectation this does add a required repetitions parameter).
2.TFQ中的基本噪声  
通过对噪声如何影响电路执行的理解，  
你可以探索噪音是如何在TFQ中工作的。  
张量流量子使用蒙特卡罗/基于轨迹的模拟作为密度矩阵模拟的替代方案。  
这是因为密度矩阵模拟的内存复杂性  
传统的全密度矩阵模拟方法限制大的模拟小于等于20个量子位。  
蒙特卡罗/轨迹法用内存成本换取额外的时间成本。  
backend='noisy'选项可用于所有tfq.layers.Sample、tfq.layers.SampledExpectation和tfq.layers.Expectation  
(对于Expection来说，这确实添加了一个必需的repetitions参数)。  

2.1 Noisy sampling in TFQ
To recreate the above plots using TFQ and trajectory simulation you can use tfq.layers.Sample
2.1 TFQ噪声采样
为了利用TFQ以及轨迹模拟技术，来重建之前的曲线，你可以使用tfq.layers.Sample
'''
"""Draw bitstring samples from `my_noisy_circuit`"""
bitstrings = tfq.layers.Sample(backend='noisy')(my_noisy_circuit, repetitions=1000)
numeric_values = np.einsum('ijk,k->ij', bitstrings.to_tensor().numpy(), [1, 2])[0]
freqs, _ = np.histogram(numeric_values, bins=[i+0.01 for i in range(-1,2** len(my_qubits))])
plt.figure(figsize=(10,5))
plt.title('Noisy Circuit Sampling')
plt.xlabel('Bitstring')
plt.ylabel('Frequency')
plt.bar([i for i in range(2** len(my_qubits))], freqs, tick_label=['00','01','10','11'])
plt.show()

'''
2.2 Noisy sample based expectation
To do noisy sample based expectation calculation you can use tfq.layers.SampleExpectation:
'''
