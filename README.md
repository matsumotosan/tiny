# tinygrad

Reimplementation of [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy, and then some.
The goal is to create a simple reimplementation of the very basic functionality of PyTorch
while remaining as simple as possible.

This serves mainly as way to familiarize myself with the inner workings of PyTorch and 
experiment with tools for testing, packaging, code coverage, and CI.

## Installation

To install tinygrad, run the following command:
```bash
pip install tinygrad
```

## Examples

```python
from micrograd.engine import Value

# Initialize variables
a = Value(1.0)

# Forward pass through simple MLP
out = MLP(x)

# Draw computational graph of MLP
draw_graph(out)
```

## Tests

All tests were run using pytest and can be run by running:
```bash
pytest
```

## References

[micrograd](https://github.com/karpathy/micrograd)
