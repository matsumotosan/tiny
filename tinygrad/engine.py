"""Tinygrad engine"""
from math import exp


class Value:
    """Scalar class to define elementary arithmetic operations and track 
    allow creation of computational graph.

    Parameters
    ----------
    data : float
        Scalar value

    _children :
        Child Value objects

    _op : str
        Operation performed to produce value

    label : str
        Value variable name
    """
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.label = label
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

        self.grad = 0.0

    def tanh(self):
        out = Value(t, (self,), 'tanh')
        x = self.data
        t = (exp(2 * x) - 1) / (exp(2 * x + 1))

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        
        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        #TODO:only int/float powers for now, implement more general case
        assert isinstance(other, (int, float))

        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        # Build topologically sorted list of value nodes
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Backpropagate gradients 
        self.grad = 1.0 # initialize gradient of leaf node
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data})"
