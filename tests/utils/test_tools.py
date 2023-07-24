import functools

from opfunu.utils.tools import alternating_array


def test_alternating_array_whenEven_thenCorrect():
    a = [-1, 1]
    expected = [-1, 1, -1, 1]
    actual = alternating_array(a, len(expected))
    assert functools.reduce(lambda x, y: x and y, map(lambda p, q: p == q, expected, actual), True)

def test_alternating_array_whenOdd_thenCorrect():
    a = [-1, 1]
    expected = [-1, 1, -1, 1, -1]
    actual = alternating_array(a, len(expected))
    assert functools.reduce(lambda x, y: x and y, map(lambda p, q: p == q, expected, actual), True)