"""Dummy test script for python-project-template."""

import pytest

from dummy import fibonacci


def test_fibonacci() -> None:
    """Test fibonacci numbers."""
    with pytest.raises(AssertionError):
        fibonacci(-1)
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(2) == 1
    assert fibonacci(3) == 2
    assert fibonacci(4) == 3
    assert fibonacci(5) == 5
    assert fibonacci(6) == 8
    assert fibonacci(7) == 13
    assert fibonacci(8) == 21
    assert fibonacci(9) == 34
