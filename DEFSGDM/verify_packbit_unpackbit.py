import torch
from torch.optim import Optimizer
import numpy as np
from timm.utils import AverageMeter
import time
from threading import Lock

from numba import njit, prange

import cupy


def np_packbits(param: np.ndarray):
    # param = np.where(param >= 0, 1, 0)
    return np.packbits(param)


def np_unpackbits(param: np.ndarray):
    param = np.unpackbits(param)
    param = np.where(param > 0, 1., -1.)
    return param


# cupy is faster than numpy in unpacking, but slower in packing,
# but I expect cupy is always faster when there isn't a powerful cpu
def cupy_packbits(param: torch.Tensor):
    # param = torch.where(param >= 0, 1, 0)
    return cupy.asnumpy(cupy.packbits(cupy.asarray(param)))


def cupy_unpackbits(param: np.ndarray):
    param = cupy.unpackbits(cupy.asarray(param))
    param = cupy.where(param > 0, 1., -1.)
    return param


# numpy seems to be faster than numba in packing bits, but slower in unpacking bits
# but I expect numba is always faster when there isn't a powerful cpu
def _numba_pack_x64(arr, su, pos):
    for i in range(64):
        j = i * 8
        su[i] = (arr[j]<<7)|(arr[j+1]<<6)|(arr[j+2]<<5)|(arr[j+3]<<4)|(arr[j+4]<<3)|(arr[j+5]<<2)|(arr[j+6]<<1)|arr[j+7]


def _numba_packbits(arr, div, su):
    # arr = np.where(arr >= 0, True, False)
    for i in range(div//64):
        _numba_pack_x64(arr[i*8:(i+64)*8], su[i:i+64], i)
    for i in range(div//64*64, div):
        j = i * 8
        su[i] = (arr[j]<<7)|(arr[j+1]<<6)|(arr[j+2]<<5)|(arr[j+3]<<4)|(arr[j+4]<<3)|(arr[j+5]<<2)|(arr[j+6]<<1)|arr[j+7]


def numba_packbits(param: torch.Tensor):
    length = param.numel()
    param = param.cpu().numpy()
    div, mod = np.divmod(length, 8)
    su = np.zeros(div + (mod > 0), dtype=np.uint8)
    _numba_packbits(param[:div*8], div, su)
    if mod > 0:
        su[-1] = sum(x*y for x, y in zip(param[div*8:], (128, 64, 32, 16, 8, 4, 2, 1)))
    return su


# numba version of unpackbits seems to be a bit faster than numpy.unpackbits
mask = 2**np.arange(7, -1, -1, dtype=np.uint8)


def numba_unpackbits(x, Nbits=8):
    out_NbitAr = np.zeros(len(x) * Nbits)
    for idx, n in enumerate(x):
        for _idx, m in enumerate(mask):
            if m & n > 0:
                out_NbitAr[idx*8 + _idx] = 1.
            else:
                out_NbitAr[idx*8 + _idx] = -1.
    return out_NbitAr


a = torch.randn(10000)
b = torch.where(a >= 0, 1, 0)
c = torch.where(a > 0, 1., -1.)

b_np = np_packbits(b.numpy())
b_cupy = cupy_packbits(b)
b_numba = np_packbits(b.numpy())

assert np.all(b_np == b_cupy)
assert np.all(b_cupy == b_numba)

c_np = np_unpackbits(b_np)
c_cupy = cupy.asnumpy(cupy_unpackbits(b_cupy))
c_numba = numba_unpackbits(b_numba)
assert np.all(c_np == c_cupy)
assert np.all(c_cupy == c_numba)