# Building

1. Download pybind11 (https://github.com/pybind/pybind11/archive/v2.5.0.tar.gz)

2. Extract pybind11

3. Compile the `cppgpu` package:

    ```bash
    nvcc \
        --compiler-options -fPIC \
        -shared -std=c++11 \
        -I/usr/local/lib/python3.5/dist-packages/numpy/core/include \
        -Ipybind11/include \
        `python3.5m-config --includes` \
        `python3.5m-config --libs` \
        -lcuda \
        cppgpu.cpp -o cppgpu`python3.5m-config --extension-suffix`
    ```

4. This will result in a file that looks like `cppgpu.cpython-35m-x86_64-linux-gnu.so`.
    
5. Run the example:

    ```bash
    python3.5 example.py
    ```
