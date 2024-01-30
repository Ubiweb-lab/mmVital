import math


def next_pow_of_2(k):
    return 2 ** math.ceil(math.log2(k))
