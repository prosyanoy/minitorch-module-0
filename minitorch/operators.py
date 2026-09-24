"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

def mul(x: float, y: float):
    return x * y


def id(x: float):
    return x


def add(x: float, y: float):
    return x + y


def neg(x: float):
    return -x


def lt(x: float, y: float):
    if x < y:
        return 1
    else:
        return 0


def eq(x: float, y: float):
    if x == y:
        return 1.0
    else:
        return 0.0


def max(x: float, y: float):
    if x > y:
        return x
    else:
        return y

def is_close(x: float, y: float):
    """
    $f(x) = |x - y| < 1e-2$
    """
    if x - y > -1e-2 and x - y < 1e-2:
        return 1.0
    else:
        return 0.0

def sigmoid(x: float):
    """
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
    """
    if x >= 0:
        value = 1.0 / (1.0 + math.exp(-x))
        if value >= 1.0:
            return math.nextafter(1.0, 0.0)
        if x > 0 and value <= 0.5:
            return math.nextafter(0.5, 1.0)
        return value

    value = math.exp(x) / (1.0 + math.exp(x))
    if value <= 0.0:
        return math.nextafter(0.0, 1.0)
    if value >= 0.5:
        return math.nextafter(0.5, 0.0)
    return value

def relu(x: float):
    if x > 0:
        return x
    else:
        return 0.0

EPS = 1e-6

def log(x: float):
    return math.log(x + EPS)


def exp(x: float):
    return math.exp(x)


def log_back(a: float, b: float):
    return b / (a + EPS)


def inv(x: float):
    return 1.0 / x


def inv_back(a: float, b: float):
    return -(1.0 / a ** 2) * b

def relu_back(x: float, y: float):
    if x > 0:
        return y
    else:
        return 0


# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def map(fn: Callable):
    def apply(my_list: Iterable[float]):
        new_list = []
        for counter, value in enumerate(my_list):
            new_list.append(fn(value))
        return new_list

    return apply

def zipWith(fn: Callable):
    def apply(my_list1: list[float], my_list2: list[float]):
        new_list = []
        for counter, value in enumerate(my_list1):
            new_list.append(fn(my_list1[counter], my_list2[counter]))
        return new_list

    return apply

def reduce(fn: Callable, start: float):
    def apply(input_list: Iterable[float]):
        my_list = list(input_list).copy()
        if len(my_list) == 0:
            return start
        current_value = my_list.pop()
        return fn(current_value, apply(my_list))

    return apply

def negList(ls: Iterable[float]):
    my_fn = map(neg)
    return my_fn(ls)

def addLists(ls1: list[float], ls2: list[float]):
    "Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`"
    return zipWith(add)(ls1, ls2)


def sum(ls: Iterable[float]):
    my_fn = reduce(add, 0)
    return my_fn(ls)

def prod(ls: Iterable[float]):
    my_fn = reduce(mul, 1)
    return my_fn(ls)
