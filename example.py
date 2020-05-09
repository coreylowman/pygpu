import numpy as np
from pygpu import on_gpu, transpile
from datetime import datetime


def t(a, b):
    return a


def mul(a, b):
    return a * b


def encrypt(a, key):
    return (a + key) % 26


def do_work(a, b):
    local = 0
    local += a + b[0]
    local += a - b[1]
    local += a * b[2]
    local += a % b[3]
    return local


def do_work2(d):
    d = d % 2
    d = d * 3
    d = d / 5
    d = d + 1
    d ^= 0x55555555
    d |= 0x77777777
    d &= 0x33333333
    d |= 0x11111111
    return d


def main():
    print(transpile(mul))
    print()

    print(transpile(encrypt))
    print()

    print(transpile(do_work))
    print()

    print(transpile(do_work2))
    print()

    # a = np.random.randn(1000000).astype(np.float32)
    # b = np.random.randn(1000000).astype(np.float32)
    a = np.array(range(10)).astype(np.float32)
    b = 2 * np.ones(10).astype(np.float32)

    # on cpu
    print('CPU')
    for i in range(5):
        start = datetime.now()
        cpu_result = mul(a, b)
        print((datetime.now() - start).total_seconds())

    # on gpu
    print('GPU')
    for i in range(5):
        start = datetime.now()
        gpu_result = on_gpu(mul, [a, b])
        print((datetime.now() - start).total_seconds())


if __name__ == '__main__':
    main()
