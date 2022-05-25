"""
qnn.py: a small quantum neural network that acts as a binary
classifier.
"""
from pyqcisim.simulator import *
from numpy.core.function_base import linspace
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import Aer, execute, QuantumCircuit
from qiskit.extensions import UnitaryGate
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import time


def convertDataToAngles(data):
    """
    Takes in a normalised 4 dimensional vector and returns
    three angles such that the encodeData function returns
    a quantum state with the same amplitudes as the
    vector passed in.
    """
    angle1 = 2 * np.arccos(np.sqrt(data[0]))
    angle2 = 2 * np.arccos(np.sqrt(data[1]))
    return np.array([angle1, angle2])


def encodeData(angles):
    """
    Given a quantum register belonging to a quantum
    circuit, performs a series of rotations and controlled
    rotations characterized by the angles parameter.
    """
    encode_c = f"""
    H qA
    H qB
    RZ qA {str(angles[0])}
    RZ qB {str(angles[1])}
    CNOT qA qB
    RZ qB {str(angles[0] * angles[1])}
    CNOT qA qB
    """
    return encode_c


# def GGate(qc, qreg, params):
#     """
#     Given a parameter α, return a single
#     qubit gate of the form
#     [cos(α), sin(α)]
#     [-sin(α), cos(α)]
#     """
#     u00 = np.cos(params[0])
#     u01 = np.sin(params[0])
#     gateLabel = "G({})".format(
#         params[0]
#     )
#     gg = """
#     RY
#     """
#     # GGate = UnitaryGate(np.array(
#     #     [[u00, u01], [-u01, u00]]
#     # ), label=gateLabel)
#     return GGate


def GLayer(params):
    """
    Applies a layer of UGates onto the qubits of register
    qreg in circuit qc, parametrized by angles params.
    """
    g_c = """"""
    denote = ['A', 'B']
    for i in range(2):
        g_c += f"""
        RY q{denote[i]} {str(-1 * 2 * params[i][0])}
        """
    return g_c


def CGLayer(params):
    """
    Applies a controlled layer of UGates, all conditioned
    on the first qubit of the anc register.
    """
    result = """"""
    denote = ['A', 'B']
    for i in range(2):
        result += f"""
        RY q{denote[i]} {str(-1 * 0.25 * params[i][0])}
        CNOT qC q{denote[i]}
        RY q{denote[i]} {str(0.5 * params[i][0])}
        CNOT qC q{denote[i]}
        RY q{denote[i]} {str(-1 * 0.25 * params[i][0])}
        """
    return result


def CXLayer(order):
    """
    Applies a layer of CX gates onto the qubits of register
    qreg in circuit qc, with the order of application
    determined by the value of the order parameter.
    """
    cx = """"""
    if order:
        cx = """
        CNOT qA qB
        """
    else:
        cx = """
        CNOT qB qA
        """
    return cx


def CCXLayer(order):
    """
    Applies a layer of Toffoli gates with the first
    control qubit always being the first qubit of the anc
    register, and the second depending on the value
    passed into the order parameter.
    """
    result = """"""
    if order:
        result += """
        H qB
        CNOT qA qB
        TD qB
        CNOT qC qB
        T qB
        CNOT qA qB
        TD qB
        CNOT qC qB
        TD qA
        T qB
        CNOT qC qA
        H qB
        TD qA
        CNOT qC qA
        T qC
        S qA
        """
        # qc.ccx(anc[0], qreg[0], qreg[1])
    else:
        result += """
                H qA
                CNOT qB qA
                TD qA
                CNOT qC qA
                T qA
                CNOT qB qA
                TD qA
                CNOT qC qA
                TD qB
                T qA
                CNOT qC qB
                H qA
                TD qB
                CNOT qC qB
                T qC
                S qB
        """
        # qc.ccx(anc[0], qreg[1], qreg[0])
    return result


def generateU(params):
    """
    Applies the unitary U(θ) to qreg by composing multiple
    U layers and CX layers. The unitary is parametrized by
    the array passed into params.
    """
    U_c = """"""
    for i in range(params.shape[0]):
        U_c += GLayer(params[i])
        U_c += CXLayer(i % 2)
    return U_c


def generateCU(qc, params):
    """
    Applies a controlled version of the unitary U(θ),
    conditioned on the first qubit of register anc.
    """
    cu = """"""
    for i in range(params.shape[0]):
        cu += CGLayer(params[i])
        cu += CCXLayer(i % 2)
    return cu


def getPrediction(qc, backend):
    """
    Returns the probability of measuring the last qubit
    in register qreg as in the |1⟩ state.
    """
    qc += """
    M qA
    """
    backend.compile(qc)
    results = pyqcisim.simulate()[1]
    backend = PyQCISim()
    if '1' in results.keys():
        return results['1'] / 1000
    else:
        return 0


def convertToClass(predictions):
    """
    Given a set of network outputs, returns class predictions
    by thresholding them.
    """
    return (predictions >= 0.5) * 1


def cost(labels, predictions):
    """
    Returns the sum of quadratic losses over the set
    (labels, predictions).
    """
    loss = 0
    for label, pred in zip(labels, predictions):
        loss += (pred - label) ** 2

    return loss / 2


def accuracy(labels, predictions):
    """
    Returns the percentage of correct predictions in the
    set (labels, predictions).
    """
    acc = 0
    for label, pred in zip(labels, predictions):
        if label == pred:
            acc += 1

    return acc / labels.shape[0]


def forwardPass(params, bias, angles, backend):
    """
    Given a parameter set params, input data in the form
    of angles, a bias, and a backend, performs a full
    forward pass on the network and returns the network
    output.
    """
    # qreg = QuantumRegister(2)
    # anc = QuantumRegister(1)
    # creg = ClassicalRegister(1)
    qc = """"""
    qc += encodeData(angles)
    qc += generateU(params)
    pred = getPrediction(qc, backend) + bias
    return pred


def computeRealExpectation(params1, params2, angles, backend):
    """
    Computes the real part of the inner product of the
    quantum states produced by acting with U(θ)
    characterised by two sets of parameters, params1 and
    params2.
    """
    qc = """"""
    qc += encodeData(angles)
    qc += """
    H qC
    """
    qc += generateCU(qc, params1)
    qc += """
    CZ qC qA
    X qC
    """
    qc += generateCU(qc, params2)
    qc += """
        X qC
        H qC
    """
    prob = getPrediction(qc, backend)
    return 2 * (prob - 0.5)


def computeGradient(params, angles, label, bias, backend):
    """
    Given network parameters params, a bias bias, input data
    angles, and a backend, returns a gradient array holding
    partials with respect to every parameter in the array
    params.
    """
    prob = forwardPass(params, bias, angles, backend)
    gradients = np.zeros_like(params)
    for i in range(params.shape[0]):
        for j in range(params.shape[1]):
            newParams = np.copy(params)
            newParams[i, j, 0] += np.pi / 2
            gradients[i, j, 0] = computeRealExpectation(
                params, newParams, angles, backend
            )
            newParams[i, j, 0] -= np.pi / 2
    biasGrad = (prob + bias - label)
    return gradients * biasGrad, biasGrad


def updateParams(params, prevParams, grads, learningRate, momentum):
    """
    Updates the network parameters using gradient descent
    and momentum.
    """
    delta = params - prevParams
    paramsNew = np.copy(params)
    paramsNew = params - grads * learningRate + momentum * delta
    return paramsNew, params


def trainNetwork(data, labels, backend):
    """
    Train a quantum neural network on inputs data and
    labels, using backend backend. Returns the parameters
    learned.
    """
    np.random.seed(1)
    trainingData = data
    test_data = test_csv[:, 0:2]
    validationData = np.array([convertDataToAngles(i) for i in test_data])
    trainingLabels = labels
    validationLabels = test_csv[:, -1]
    params = np.random.sample((3, 2, 1))
    bias = 0.01
    prevParams = np.copy(params)
    prevBias = bias
    batchSize = 5
    iterationnum = int(labels.shape[0] / batchSize)
    momentum = 0.9
    good_params = []
    loss = []
    good_validation = 0
    good_bias = 0.01
    for epoch in range(1, 3):
        learningRate = 0.07 * np.exp(-epoch / 3)
        print(f'epoch:{epoch}')
        for iteration in range(iterationnum):
            samplePos = iteration * batchSize
            batchTrainingData = trainingData[samplePos:samplePos + batchSize]
            batchLabels = trainingLabels[samplePos:samplePos + batchSize]
            batchGrads = np.zeros_like(params)
            batchBiasGrad = 0
            for i in range(batchSize):
                grads, biasGrad = computeGradient(
                    params, batchTrainingData[i], batchLabels[i], bias, backend
                )
                batchGrads += grads / batchSize
                batchBiasGrad += biasGrad / batchSize

            params, prevParams = updateParams(
                params, prevParams, batchGrads, learningRate, momentum
            )

            temp = bias
            bias += -learningRate * batchBiasGrad + momentum * (bias - prevBias)
            prevBias = temp

            trainingPreds = np.array([forwardPass(
                params, bias, angles, backend
            ) for angles in trainingData])
            costvalue = cost(trainingLabels, trainingPreds)
            print('Iteration {} | Loss: {}'.format(
                iteration + 1, costvalue
            ))
            loss.append(costvalue)

        validationProbs = np.array(
            [forwardPass(
                params, bias, angles, backend
            ) for angles in validationData]
        )
        validationClasses = convertToClass(validationProbs)
        validationAcc = accuracy(validationLabels, validationClasses)
        print('Validation accuracy:', validationAcc)
        for x, y, p in zip(validationData, validationLabels, validationClasses):
            print('Data:', x, ' | Class:', y, ' | Prediction:', p)
        if good_validation <= validationAcc:
            good_params = params
            good_validation = validationAcc
            good_bias = bias
            print('update params:')
            print(f'bias: {good_bias}')
            print(good_params)
        else:
            print('not update params:')
            print(f'bias: {good_bias}')
            print(good_params)

    result = np.array(list(zip(test_csv[:, 0], test_csv[:, 1], validationClasses)))
    re = DataFrame(result)
    los = DataFrame(list(zip([i + 1 for i in range(int(iterationnum * epoch))],loss)))
    re.to_csv("training_result.csv", index=False, header=False)
    los.to_csv("loss.csv", index=False, header=False)

    return result


"""read training and testing data set"""
data = np.genfromtxt("training_set.csv", delimiter=",")
test_csv = np.genfromtxt("testing_set.csv", delimiter=",")
X = data[:, 0:2]
Y = data[:, -1]


"""main program"""
tic1 = time.perf_counter()
features = np.array([convertDataToAngles(i) for i in X])
pyqcisim = PyQCISim()
result = trainNetwork(features, Y, pyqcisim)
toc1 = time.perf_counter()
print(f'time consumption: {toc1-tic1}s')


"""draw testing result"""
x1 = []
y1 = []
x0 = []
y0 = []
for i in range(int(test_csv.shape[0])):
    if result[i, -1] == 1:
        x1.append(result[i, 0])
        y1.append(result[i, 1])
    else:
        x0.append(result[i, 0])
        y0.append(result[i, 1])
fig, ax = plt.subplots()
ax.set_aspect(1)
plt.title('testing result', fontsize=22)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.xlim(-0.3, 1.3)
plt.ylim(-0.3, 1.3)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.scatter(x1, y1, label="label: 1")
plt.scatter(x0, y0, label="label: 0")
plt.legend(loc=0, fontsize=12)
plt.show()