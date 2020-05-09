from .transpiler import transpile
import subprocess
import numpy as np


class Context(object):
    def __init__(self):
        self._initialized = False
        self._cu_by_fn = {}
        self._ptx_by_fn = {}
        self._cpp_ctx = None

    def init(self):
        """
        Initializes CUDA devices and context.
        """
        from cppgpu import CppContext

        if not self._initialized:
            self._initialized = True
            self._cpp_ctx = CppContext()
            self._cpp_ctx.init()

    def compile(self, fn, debug=False):
        """
        Does the following:

            1. Transpiles the function into a CUDA kernel (see transpiler.py)
            2. Writes it to a .cu file
            3. Compiles it to a .ptx file using nvcc
            4. Loads it in c++ with cuModuleLoad and cuModuleGetFunction (see cppgpu.cpp)

        :param fn: A python function object
        :param debug: whether to print the kernel source
        """
        # if we already compiled this function, don't do it again
        if fn in self._ptx_by_fn:
            return

        # transpile to C++
        fn_source = transpile(fn)
        if debug:
            print(fn_source)

        # write to a .cu file
        self._cu_by_fn[fn] = '.pygpu_{}.cu'.format(fn.__name__)
        with open(self._cu_by_fn[fn], 'w') as fp:
            fp.write(fn_source)

        # compile to a .ptx file
        self._ptx_by_fn[fn] = '.pygpu_{}.ptx'.format(fn.__name__)
        subprocess.check_call(['nvcc', '-ptx', self._cu_by_fn[fn], '-o', self._ptx_by_fn[fn]])

        self._cpp_ctx.store_function(fn.__name__, self._ptx_by_fn[fn])

    def invoke(self, fn, params, blocks_per_grid, threads_per_block):
        """
        Call a function on the GPU. Does the following:

            1. Allocates device memory for parameters & the result
            2. Copies host memory to device memory
            3. Launches the kernel created for the function
            4. Copies the result to host memory

        See cppgpu.cpp for all the functions.

        :param fn: A python function object
        :param params: the parameters for the function
        :param blocks_per_grid: number of blocks per grid
        :param threads_per_block: number of threads per block
        :return: numpy.ndarray with the result
        """
        # initialize host memory for result
        result = np.zeros_like(params[0])

        self._cpp_ctx.clear_params()

        # allocate device memory for each parameter & result
        for i, param in enumerate(params):
            self._cpp_ctx.alloc_param(i, param)
        self._cpp_ctx.alloc_param(len(params), result)

        # copy host memory to device for each parameter & result
        for i, param in enumerate(params):
            self._cpp_ctx.param_to_device(i, param)
        self._cpp_ctx.param_to_device(len(params), result)

        self._cpp_ctx.launch_kernel(fn.__name__, blocks_per_grid, threads_per_block)

        self._cpp_ctx.param_from_device(len(params), result)

        return result
