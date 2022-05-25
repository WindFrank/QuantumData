"""Minimal example of defining a custom channel."""
import cirq
# 这是一个信道噪声的Custom Channel，即自定义噪声的实例。如果想要自己定义噪声，用下面这个类似的模板


class BitAndPhaseFlipChannel(cirq.SingleQubitGate):
    # 这个方法是初始化方法，传入的参数就是发生概率
    def __init__(self, p: float) -> None:
        self._p = p

    # 该方法定义了信道噪声的发生概率ps和对应的矩阵操作
    def _mixture_(self):
        ps = [1.0 - self._p, self._p]
        ops = [cirq.unitary(cirq.I), cirq.unitary(cirq.Y)]
        return tuple(zip(ps, ops))

    # 这个方法是判断是否可混合的方法，可以在不必要时不重现
    def _has_mixture_(self) -> bool:
        return True

    # 这个方法应该是绘图用的
    def _circuit_diagram_info_(self, args) -> str:
        return f"BitAndPhaseFlip({self._p})"


'''
Note: If a custom channel does not have a mixture, 
it should instead define the _kraus_ magic method to return a sequence of Kraus operators (as numpy.ndarrays). 
Defining a _has_kraus_ method which returns True is optional but recommended.
'''


