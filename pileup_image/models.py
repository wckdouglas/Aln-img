from enum import Enum

TRUNCATED_READ_DEPTH = 50


class Matrix(Enum):
    """
    Enum encoding the identity of the matrix
    """

    FORWARD_BASE = 0
    REVERSE_BASE = 1
    INSERTION = 2
    DELETION = 3


class Nucleotide(Enum):
    """
    Enum encoding the acceptable bases
    """

    A = 0
    C = 1
    G = 2
    T = 3
    N = 4
