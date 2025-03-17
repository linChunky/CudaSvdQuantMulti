# libsvd_quant_multi

## Introduction
`libsvd_quant_multi` is a CUDA-based library that computes the Singular Value Decomposition (SVD) of a given matrix and performs quantization on the decomposed matrices. The implementation utilizes `cuSolver` and `cuBLAS` for efficient computation on the GPU.

### Features
- Computes SVD for a given matrix stored in column-major order.
- Constructs `U_sqrt`, `sqrtVT`, and `A_reconstructed` from the decomposition.
- Supports block quantization of `U_sqrt` and `sqrtVT` for `INT16`, `INT8`, and `INT4` formats.
- Provides a Python interface for easy integration and testing.
- Includes a GPU-accelerated dequantization and matrix multiplication function.

## Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit (>= 11.0)
- `cuBLAS` and `cuSolver` libraries
- Python (>= 3.7)
- `numpy`

## Compilation
To compile the shared library on Linux:
```sh
make
```

## Usage
### C++ Interface
```cpp
extern "C" SVDResults* libsvd_quant_multi_interface(const float *h_A, int m, int n, int quant_type);
extern "C" void free_svd_results(SVDResults* res);
extern "C" float* dequantized_matmul_interface(
    void* quantized_U, float* U_scales, int U_rows, int U_cols,
    void* quantized_VT, float* VT_scales, int VT_rows, int VT_cols,
    int quant_block_rows, int quant_block_cols, int quant_type);
```

### Python Interface
```python
import ctypes
import numpy as np

# Load shared library
lib = ctypes.cdll.LoadLibrary('./libsvd_quant_multi.so')

# Example usage
m, n = 4096, 4096
A = np.random.rand(m, n).astype(np.float32, order='F')
quant_type = 8  # Choose from 16, 8, 4
res_ptr = lib.svd_test_interface(A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), m, n, quant_type)
res = res_ptr.contents

# Free memory
lib.free_svd_results(res)
```

---

# libsvd_quant_multi

## 简介
`libsvd_quant_multi` 是一个基于 CUDA 的库，用于计算给定矩阵的奇异值分解（SVD），并对分解后的矩阵进行量化处理。本项目利用 `cuSolver` 和 `cuBLAS` 进行高效计算。

### 功能
- 计算存储为列主序的矩阵的 SVD。
- 构造 `U_sqrt`、`sqrtVT` 和 `A_reconstructed`。
- 支持 `INT16`、`INT8` 和 `INT4` 格式的块量化。
- 提供 Python 接口，方便集成与测试。
- 包含 GPU 端的反量化与矩阵乘法函数。

## 依赖
- 具备 CUDA 支持的 NVIDIA GPU
- CUDA Toolkit（>= 11.0）
- `cuBLAS` 和 `cuSolver` 库
- Python（>= 3.7）
- `numpy`

## 编译
在 Linux 上编译共享库：
```sh
make
```

## 使用
### C++ 接口
```cpp
extern "C" SVDResults* libsvd_quant_multi_interface(const float *h_A, int m, int n, int quant_type);
extern "C" void free_svd_results(SVDResults* res);
extern "C" float* dequantized_matmul_interface(
    void* quantized_U, float* U_scales, int U_rows, int U_cols,
    void* quantized_VT, float* VT_scales, int VT_rows, int VT_cols,
    int quant_block_rows, int quant_block_cols, int quant_type);
```

### Python 接口
```python
import ctypes
import numpy as np

# 加载共享库
lib = ctypes.cdll.LoadLibrary('./libsvd_quant_multi.so')

# 示例
m, n = 4096, 4096
A = np.random.rand(m, n).astype(np.float32, order='F')
quant_type = 8  # 可选 16, 8, 4
res_ptr = lib.svd_test_interface(A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), m, n, quant_type)
res = res_ptr.contents

# 释放内存
lib.free_svd_results(res)
```


