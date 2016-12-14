import functools
import itertools


def debug(function):

    def wrapper(*args, **kwargs):
        print(function)
        return function(*args, **kwargs)

    return wrapper


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


class FunctionChain(object):

    def __init__(self, *args):
        self._functions = args
        self._permutations = list(itertools.permutations(args))
        self._count = 0

    def __len__(self):
        return len(self._permutations)

    @property
    def count(self):
        return self._count

    def get(self, index=None):
        if index is None:
            i = self._count
        else:
            i = index
        self._count += 1
        i %= self.__len__()
        funs = self._permutations[i]
        return compose(*funs)

