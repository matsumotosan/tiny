"""Test basic arithmetic operators for Value objects."""
import pytest

from tinygrad.engine import Value


def test_add():
    """Test add and radd."""
    a = Value(1.0) 
    b = Value(2.0) 
    c = a + b
    d = a + 3.0
    e = 3.0 + b
    
    assert c.data == 3.0
    assert d.data == 4.0
    assert e.data == 5.0


def test_sub():
    """Test sub and rsub."""
    a = Value(1.0) 
    b = Value(2.0) 
    c = a - b
    d = a - 3.0
    e = 3.0 - b
    
    assert c.data == -1.0
    assert d.data == -2.0
    assert e.data == 1.0


def test_mul():
    """Test mul and rmul."""
    a = Value(1.0) 
    b = Value(2.0) 
    c = a * b
    d = a * 3.0
    e = 3.0 * b
    
    assert c.data == 2.0
    assert d.data == 3.0
    assert e.data == 6.0


def test_div():
    """Test div and rdiv."""
    a = Value(1.0) 
    b = Value(2.0) 
    c = a / b
    d = a / 3.0
    e = 3.0 / b
    
    assert c.data == 0.5
    assert d.data == 1.0 / 3.0
    assert e.data == 1.5


def test_relu():
    """Test ReLU."""
    a = Value(1.0)
    b = Value(-1.0)

    assert a.relu().data == 1.0
    assert b.relu().data == 0.0
