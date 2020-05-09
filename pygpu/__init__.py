import numpy as np
from .context import Context
from .transpiler import transpile

"""
Global Context object
"""
ctx = Context()


def on_gpu(fn, params, blocks_per_grid=None, threads_per_block=1, debug=False):
    """
    Invoke the function passed in on the GPU.

    :param fn: A python function object
    :param params: The parameters to pass to the function - should all be numpy arrays
    :param blocks_per_grid: default is `(N + threads_per_block - 1) // threads_per_block`
    :param threads_per_block: default is 1
    :param debug: whether to print the transpiled function's kernel source
    :return: numpy.ndarray
    """
    if blocks_per_grid is None:
        blocks_per_grid = (params[0].shape[0] + threads_per_block - 1) // threads_per_block

    ctx.init()
    ctx.compile(fn, debug=debug)
    result = ctx.invoke(fn, params, blocks_per_grid=blocks_per_grid, threads_per_block=threads_per_block)
    return result
