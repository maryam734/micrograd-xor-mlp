class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build_topo(child)

                topo.append(v)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


# Testing
a = Value(2.0)
b = Value(3.0)

c = a * b
d = c + a

d.backward()

print("Output d =", d.data)
print("Gradient of a =", a.grad)
print("Gradient of b =", b.grad)