#include <iostream>
#include <cstdlib>
#include <map>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <numpy/arrayobject.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class CppContext {
protected:
    CUdevice _device;
    CUcontext _context;

    std::map<std::string, CUmodule> _module_by_name;
    std::map<std::string, CUfunction> _function_by_name;

    std::vector<CUdeviceptr> _params;

public:
    CppContext() {}
    ~CppContext() {}

    void init() {
        cuInit(0);
        cuDeviceGet(&_device, 0);
        cuCtxCreate(&_context, 0, _device);
    }

    void store_function(std::string name, std::string module_path) {
        CUmodule module;
        cuModuleLoad(&module, module_path.c_str());

        CUfunction function;
        cuModuleGetFunction(&function, module, name.c_str());

        _module_by_name[name] = module;
        _function_by_name[name] = function;
    }

    void clear_params() {
        for (size_t i = 0;i < _params.size();i++) {
            cuMemFree(_params[i]);
        }
        _params.clear();
    }

    void alloc_param(size_t i, py::array_t<float> param) {
        CUdeviceptr device_param;
        cuMemAlloc(&device_param, param.size() * sizeof(float));
        _params.push_back(device_param);
    }

    void param_to_device(size_t i, py::array_t<float> param) {
        cuMemcpyHtoD(_params[i], param.data(), param.size() * sizeof(float));
    }

    void param_from_device(size_t i, py::array_t<float> &param) {
        cuMemcpyDtoH(param.mutable_data(), _params[i], param.size() * sizeof(float));
    }

    void launch_kernel(std::string name, int blocks_per_grid, int threads_per_block) {
        std::vector<CUdeviceptr *> args;
        for (size_t i = 0;i < _params.size();i++) {
            args.push_back(&_params[i]);
        }
        cuLaunchKernel(
            _function_by_name[name],
            blocks_per_grid, 1, 1,
            threads_per_block, 1, 1,
            0, 0, (void **)args.data(), 0
        );
        cuCtxSynchronize();
    }
};

PYBIND11_MODULE(cppgpu, m) {
    py::class_<CppContext>(m, "CppContext")
        .def(py::init<>())
        .def("init", &CppContext::init)
        .def("store_function", &CppContext::store_function)
        .def("clear_params", &CppContext::clear_params)
        .def("alloc_param", &CppContext::alloc_param)
        .def("param_to_device", &CppContext::param_to_device)
        .def("param_from_device", &CppContext::param_from_device)
        .def("launch_kernel", &CppContext::launch_kernel);
}
