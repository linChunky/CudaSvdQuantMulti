# svd_test.py
import ctypes
import numpy as np
import os

class SVDResults(ctypes.Structure):
    _fields_ = [
        ("m", ctypes.c_int),
        ("n", ctypes.c_int),
        ("k", ctypes.c_int),
        ("S", ctypes.POINTER(ctypes.c_float)),           # length k
        ("U", ctypes.POINTER(ctypes.c_float)),           # m×k
        ("VT", ctypes.POINTER(ctypes.c_float)),          # k×n
        ("U_sqrt", ctypes.POINTER(ctypes.c_float)),      # m×k
        ("sqrtVT", ctypes.POINTER(ctypes.c_float)),      # k×n
        ("A_reconstructed", ctypes.POINTER(ctypes.c_float)),# m×n
        ("quant_type", ctypes.c_int),
        ("quantized_U_sqrt", ctypes.c_void_p),
        ("U_sqrt_scales", ctypes.POINTER(ctypes.c_float)),  # length = num_blocks（U_sqrt）
        ("quantized_sqrtVT", ctypes.c_void_p), 
        ("sqrtVT_scales", ctypes.POINTER(ctypes.c_float)),
        ("quant_block_rows", ctypes.c_int),
        ("quant_block_cols", ctypes.c_int),
        ("U_sqrt_num_blocks", ctypes.c_int),
        ("sqrtVT_num_blocks", ctypes.c_int)
    ]

# 加载共享库（请确保 libsvd_test.so 与本脚本在同一目录下，或设置正确路径）
lib_path = os.path.join(os.path.dirname(__file__), 'libsvd_quant_multi.so')
lib = ctypes.cdll.LoadLibrary(lib_path)

# 设置 SVD 接口函数原型：增加 quant_type 参数
lib.svd_test_interface.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.svd_test_interface.restype = ctypes.POINTER(SVDResults)

lib.free_svd_results.argtypes = [ctypes.POINTER(SVDResults)]
lib.free_svd_results.restype = None

# 设置 GPU 侧反量化乘法接口原型
lib.dequantized_matmul_interface.argtypes = [
    ctypes.c_void_p,                 # quantized_U
    ctypes.POINTER(ctypes.c_float),  # U_scales
    ctypes.c_int, ctypes.c_int,      # U_rows, U_cols
    ctypes.c_void_p,                 # quantized_VT
    ctypes.POINTER(ctypes.c_float),  # VT_scales
    ctypes.c_int, ctypes.c_int,      # VT_rows, VT_cols
    ctypes.c_int, ctypes.c_int,      # quant_block_rows, quant_block_cols
    ctypes.c_int                     # quant_type
]
lib.dequantized_matmul_interface.restype = ctypes.POINTER(ctypes.c_float)

# 辅助函数：将 ctypes 指针转换为 NumPy 数组（先转换为一维，再 reshape 为 Fortran-order）
def ptr_to_array(ptr, total_size, shape):
    buf = np.ctypeslib.as_array(ptr, shape=(total_size,))
    return np.reshape(buf, shape, order='F')

if __name__ == '__main__':
    # Input: Demensions & Array(float32)
    m, n = 4096, 4096
    A = np.random.rand(m, n).astype(np.float32, order='F')

    # Quant Type Int16 Int8 Int4 supported
    for qt in [16, 8, 4]:
        print(f"\n================== Test Quant Type INT{qt} ==================")
        res_ptr = lib.svd_test_interface(A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), m, n, qt)
        res = res_ptr.contents
        deq_product_ptr = lib.dequantized_matmul_interface(
            res.quantized_U_sqrt,
            res.U_sqrt_scales,
            m, res.k,
            res.quantized_sqrtVT,
            res.sqrtVT_scales,
            res.k, n,
            res.quant_block_rows,
            res.quant_block_cols,
            res.quant_type
        )
        # print the dequant res
        print(ptr_to_array(deq_product_ptr, m*n, (m, n)))
        lib.free_svd_results(res)
