import numpy as np


def alternating_array(a, n):
    # This method is meant for alternating values in a.
    if len(a) != 2:
        raise ValueError('a should be of len 2')
    # Create an array of alternating values
    alternating = np.tile(a, n // 2)

    # If n is odd, add an element to the end
    if n % 2 == 1:
        alternating = np.append(alternating, a[0])

    return alternating
