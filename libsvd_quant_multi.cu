// libsvd_quant_multi.cu
//
// 编译共享库示例（Linux）：
// nvcc -shared -Xcompiler -fPIC -o liblibsvd_quant_multi.so libsvd_quant_multi.cu -lcusolver -lcublas
//
// 本文件实现：
// 1. 根据外部传入的矩阵（要求列主序存储）计算 SVD，并构造 U_sqrt、sqrtVT、A_reconstructed 等。
// 2. 根据额外的参数 quant_type（16/8/4）对 U_sqrt 与 sqrtVT 进行分块量化，量化后同时返回量化数据和每块的 scale 数组。
//
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#define QUANT_BLOCK_ROWS 32
#define QUANT_BLOCK_COLS 32

// 错误检查宏
#define CUDA_CHECK(err) do { \
    if((err) != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUSOLVER_CHECK(err) do { \
    if((err) != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "CUSOLVER Error at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUBLAS_CHECK(err) do { \
    if((err) != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "CUBLAS Error at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

//-----------------------------------------------------------------
// Kernel：计算 C = A * diag(d)
// A 为 m×n（列主序存储），对每一列 j，C(:,j)=A(:,j)*d[j]
__global__ void diagRightMul(const float *A, const float *d, float *C, int m, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n) {
        C[row + col*m] = A[row + col*m] * d[col];
    }
}

// Kernel：计算 C = diag(d) * A
// A 为 m×n（列主序存储），对每一行 i，C(i,:)=d[i]*A(i,:)
__global__ void diagLeftMul(const float *A, const float *d, float *C, int m, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n) {
        C[row + col*m] = d[row] * A[row + col*m];
    }
}

template <typename T>
__global__ void block_quantize(const float* in, T* out, float* scales,
                               int rows, int cols,
                               int block_rows, int block_cols,
                               float q_max)
{
    // 计算当前分块的起始行列
    int blockRow = blockIdx.x;
    int blockCol = blockIdx.y;
    int row_start = blockRow * block_rows;
    int col_start = blockCol * block_cols;

    // 每个线程在分块内的局部坐标
    int thread_row = threadIdx.x;
    int thread_col = threadIdx.y;
    int global_row = row_start + thread_row;
    int global_col = col_start + thread_col;
    
    // 每个线程从全局内存中加载数据到局部寄存器（如果在范围内）
    float orig_val = 0.0f;
    if(global_row < rows && global_col < cols)
        orig_val = in[global_row + global_col * rows];
    // 计算绝对值，用于归约
    float my_abs = fabsf(orig_val);
    
    // 使用共享内存归约，计算当前块内的最大绝对值
    extern __shared__ float sdata[];
    int tid = thread_row * blockDim.y + thread_col;
    sdata[tid] = my_abs;
    __syncthreads();
    
    int blockSize = blockDim.x * blockDim.y;
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if(tid < s) {
            // 归约取最大值
            if(sdata[tid] < sdata[tid + s])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 计算当前块在整个矩阵中的块索引
    int gridX = gridDim.x; // 横向块数
    int block_index = blockRow + blockCol * gridX;
    if (tid == 0)
        scales[block_index] = sdata[0];
    __syncthreads();
    
    // 每个线程利用归约得到的 scale 进行量化
    float scale = scales[block_index];
    if(global_row < rows && global_col < cols) {
        T qval;
        if(scale == 0)
            qval = 0;
        else {
            float normalized = orig_val / scale;  // 直接使用局部变量 orig_val
            float q = normalized * q_max;
            int q_int = (int)roundf(q);
            if(q_int > (int)q_max) q_int = (int)q_max;
            if(q_int < -(int)q_max) q_int = -(int)q_max;
            qval = (T)q_int;
        }
        out[global_row + global_col * rows] = qval;
    }
}


//-----------------------------------------------------------------
// 定义 SVD 结果结构体，所有返回数组均在 host 内存中
typedef struct {
    int m;                // 原矩阵行数
    int n;                // 原矩阵列数
    int k;                // k = min(m,n)
    float* S;             // 奇异值向量，长度 k
    float* U;             // U_k，尺寸 m×k（全 U 的前 k 列）
    float* VT;            // VT_k，尺寸 k×n（全 VT 的前 k 行）
    float* U_sqrt;        // U_sqrt = U_k * diag(sqrt(S))，尺寸 m×k
    float* sqrtVT;        // sqrtVT = diag(sqrt(S)) * VT_k，尺寸 k×n
    float* A_reconstructed; // A_reconstructed = U_sqrt * sqrtVT，尺寸 m×n

    // 以下为量化结果（若量化标志非0，则有效）
    int quant_type;       // 16/8/4 表示 INT16/INT8/INT4
    void* quantized_U_sqrt; // 量化后的 U_sqrt，尺寸 m×k（host端，类型 short 或 char）
    float* U_sqrt_scales;   // U_sqrt 分块量化时每块的 scale，数组长度 = num_blocks_U
    void* quantized_sqrtVT; // 量化后的 sqrtVT，尺寸 k×n
    float* sqrtVT_scales;   // sqrtVT 分块量化时每块的 scale，数组长度 = num_blocks_VT
    int quant_block_rows;   // 分块量化时的块行数（例如32）
    int quant_block_cols;   // 分块量化时的块列数（例如32）
    int U_sqrt_num_blocks;  // U_sqrt 总块数
    int sqrtVT_num_blocks;  // sqrtVT 总块数
} SVDResults;

//-----------------------------------------------------------------
// 接口函数：计算 SVD、构造 U_sqrt、sqrtVT、A_reconstructed，并根据 quant_type 对 U_sqrt 与 sqrtVT 进行量化
// 输入 h_A 要求按列主序存储
// quant_type 取值 16/8/4，若传入其他值则不做量化（对应字段置为NULL）
extern "C" SVDResults* libsvd_quant_multi_interface(const float *h_A, int m, int n, int quant_type)
{
    int k = (m < n ? m : n);
    const int lda = m;  // 列主序

    // 1. 将输入矩阵拷贝到 device
    float *d_A = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeof(float)*m*n));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeof(float)*m*n, cudaMemcpyHostToDevice));

    // 2. 创建 cuSOLVER handle
    cusolverDnHandle_t cusolverH = nullptr;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    // 3. 分配 device 内存存放 SVD 结果：S（长度 k）、U（m×m）和 VT（n×n）
    float *d_S = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_S, sizeof(float)*k));
    float *d_U = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_U, sizeof(float)*m*m));
    float *d_VT = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_VT, sizeof(float)*n*n));

    // 4. 查询 workspace 大小，并分配 workspace 与 devInfo
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(cusolverH, m, n, &lwork));
    float *d_work = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_work, sizeof(float)*lwork));
    int *devInfo = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));

    // 5. SVD 分解参数：计算全 U 与 VT
    signed char jobu = 'A';
    signed char jobvt = 'A';
    CUSOLVER_CHECK(cusolverDnSgesvd(
        cusolverH,
        jobu, jobvt,
        m, n,
        d_A, lda,
        d_S,
        d_U, lda,    // d_U 为 m×m
        d_VT, n,     // d_VT 为 n×n
        d_work, lwork,
        nullptr,
        devInfo));

    int info_gpu = 0;
    CUDA_CHECK(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if(info_gpu != 0) {
        fprintf(stderr, "Error: SVD failed, info = %d\n", info_gpu);
        exit(EXIT_FAILURE);
    }

    // 6. 从 d_U 提取 U_k（前 k 列，m×k）
    float* h_U = (float*)malloc(sizeof(float)*m*k);
    CUDA_CHECK(cudaMemcpy(h_U, d_U, sizeof(float)*m*k, cudaMemcpyDeviceToHost));

    // 7. 从 d_VT 提取 VT_k（前 k 行，尺寸 k×n）
    float* h_VT = (float*)malloc(sizeof(float)*k*n);
    for (int j = 0; j < n; j++) {
        CUDA_CHECK(cudaMemcpy(h_VT + j*k, d_VT + j*n, sizeof(float)*k, cudaMemcpyDeviceToHost));
    }

    // 8. 将奇异值 S 拷贝到 host
    float* h_S = (float*)malloc(sizeof(float)*k);
    CUDA_CHECK(cudaMemcpy(h_S, d_S, sizeof(float)*k, cudaMemcpyDeviceToHost));

    // 9. 计算 sqrt(S)（在 host 上）
    float* h_S_sqrt = (float*)malloc(sizeof(float)*k);
    for (int i = 0; i < k; i++) {
        h_S_sqrt[i] = sqrtf(h_S[i]);
    }
    // 同时拷贝 sqrt(S)到 device
    float *d_S_sqrt = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_S_sqrt, sizeof(float)*k));
    CUDA_CHECK(cudaMemcpy(d_S_sqrt, h_S_sqrt, sizeof(float)*k, cudaMemcpyHostToDevice));

    // 10. 计算 U_sqrt = U_k * diag(sqrt(S)) （m×k）
    float *d_U_k;
    CUDA_CHECK(cudaMalloc((void**)&d_U_k, sizeof(float)*m*k));
    CUDA_CHECK(cudaMemcpy(d_U_k, d_U, sizeof(float)*m*k, cudaMemcpyDeviceToDevice));
    float *d_U_sqrt;
    CUDA_CHECK(cudaMalloc((void**)&d_U_sqrt, sizeof(float)*m*k));
    {
        dim3 threads(16,16);
        dim3 blocks((m+15)/16, (k+15)/16);
        // 注意：使用 d_S_sqrt 而非 d_S
        diagRightMul<<<blocks, threads>>>(d_U_k, d_S_sqrt, d_U_sqrt, m, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    float* h_U_sqrt = (float*)malloc(sizeof(float)*m*k);
    CUDA_CHECK(cudaMemcpy(h_U_sqrt, d_U_sqrt, sizeof(float)*m*k, cudaMemcpyDeviceToHost));

    // 11. 计算 sqrtVT = diag(sqrt(S)) * VT_k （k×n）
    float *d_VT_k;
    CUDA_CHECK(cudaMalloc((void**)&d_VT_k, sizeof(float)*k*n));
    CUDA_CHECK(cudaMemcpy(d_VT_k, h_VT, sizeof(float)*k*n, cudaMemcpyHostToDevice));
    float *d_sqrtVT;
    CUDA_CHECK(cudaMalloc((void**)&d_sqrtVT, sizeof(float)*k*n));
    {
        dim3 threads(16,16);
        dim3 blocks((k+15)/16, (n+15)/16);
        // 注意：使用 d_S_sqrt
        diagLeftMul<<<blocks, threads>>>(d_VT_k, d_S_sqrt, d_sqrtVT, k, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    float* h_sqrtVT = (float*)malloc(sizeof(float)*k*n);
    CUDA_CHECK(cudaMemcpy(h_sqrtVT, d_sqrtVT, sizeof(float)*k*n, cudaMemcpyDeviceToHost));

    // 12. 计算 A_reconstructed = U_sqrt * sqrtVT （m×n），使用 cuBLAS
    float *d_A_reconstructed;
    CUDA_CHECK(cudaMalloc((void**)&d_A_reconstructed, sizeof(float)*m*n));
    cublasHandle_t cublasH = nullptr;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cublasH,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, k,
                             &alpha,
                             d_U_sqrt, m,
                             d_sqrtVT, k,
                             &beta,
                             d_A_reconstructed, m));
    float* h_A_reconstructed = (float*)malloc(sizeof(float)*m*n);
    CUDA_CHECK(cudaMemcpy(h_A_reconstructed, d_A_reconstructed, sizeof(float)*m*n, cudaMemcpyDeviceToHost));

    // 13. 填充 SVDResults 结构体（所有数组在 host）
    SVDResults* results = (SVDResults*)malloc(sizeof(SVDResults));
    results->m = m;
    results->n = n;
    results->k = k;
    results->S = h_S;
    results->U = h_U;
    results->VT = h_VT;
    results->U_sqrt = h_U_sqrt;
    results->sqrtVT = h_sqrtVT;
    results->A_reconstructed = h_A_reconstructed;

    // 默认量化字段置空
    results->quant_type = 0;
    results->quantized_U_sqrt = NULL;
    results->U_sqrt_scales = NULL;
    results->quantized_sqrtVT = NULL;
    results->sqrtVT_scales = NULL;
    results->quant_block_rows = 0;
    results->quant_block_cols = 0;
    results->U_sqrt_num_blocks = 0;
    results->sqrtVT_num_blocks = 0;

    // 14. 根据 quant_type 对 U_sqrt 和 sqrtVT 进行量化（分块量化）
    if (quant_type == 16 || quant_type == 8 || quant_type == 4) {
        results->quant_type = quant_type;
        int quant_block_rows = QUANT_BLOCK_ROWS;
        int quant_block_cols = QUANT_BLOCK_COLS;
        results->quant_block_rows = quant_block_rows;
        results->quant_block_cols = quant_block_cols;
        // 对 U_sqrt（尺寸 m×k）分块
        int U_blocks_row = (m + quant_block_rows - 1) / quant_block_rows;
        int U_blocks_col = (k + quant_block_cols - 1) / quant_block_cols;
        int U_num_blocks = U_blocks_row * U_blocks_col;
        results->U_sqrt_num_blocks = U_num_blocks;
        // 对 sqrtVT（尺寸 k×n）分块
        int VT_blocks_row = (k + quant_block_rows - 1) / quant_block_rows;
        int VT_blocks_col = (n + quant_block_cols - 1) / quant_block_cols;
        int VT_num_blocks = VT_blocks_row * VT_blocks_col;
        results->sqrtVT_num_blocks = VT_num_blocks;
        
        // 根据 quant_type 决定输出数据类型与 q_max
        float q_max = 0.0f;
        if (quant_type == 16) { q_max = 32767.0f; }
        else if (quant_type == 8) { q_max = 127.0f; }
        else if (quant_type == 4) { q_max = 7.0f; }
        
        // 分配设备内存用于量化结果和块 scale
        void* d_quantized_U_ptr = NULL;
        if (quant_type == 16) {
            short* tmp;
            CUDA_CHECK(cudaMalloc((void**)&tmp, sizeof(short)*m*k));
            d_quantized_U_ptr = tmp;
        } else {
            char* tmp;
            CUDA_CHECK(cudaMalloc((void**)&tmp, sizeof(char)*m*k));
            d_quantized_U_ptr = tmp;
        }
        float* d_U_scales;
        CUDA_CHECK(cudaMalloc((void**)&d_U_scales, sizeof(float)*U_num_blocks));
        
        void* d_quantized_VT_ptr = NULL;
        if (quant_type == 16) {
            short* tmp;
            CUDA_CHECK(cudaMalloc((void**)&tmp, sizeof(short)*k*n));
            d_quantized_VT_ptr = tmp;
        } else {
            char* tmp;
            CUDA_CHECK(cudaMalloc((void**)&tmp, sizeof(char)*k*n));
            d_quantized_VT_ptr = tmp;
        }
        float* d_VT_scales;
        CUDA_CHECK(cudaMalloc((void**)&d_VT_scales, sizeof(float)*VT_num_blocks));
        
        // 设置核函数的块尺寸和网格尺寸
        dim3 quantBlock(quant_block_rows, quant_block_cols);
        dim3 gridU((m + quant_block_rows - 1)/quant_block_rows, (k + quant_block_cols - 1)/quant_block_cols);
        dim3 gridVT((k + quant_block_rows - 1)/quant_block_rows, (n + quant_block_cols - 1)/quant_block_cols);
        size_t sharedMemSize = quant_block_rows * quant_block_cols * sizeof(float);
        
        // 对 U_sqrt 量化
        if (quant_type == 16) {
            block_quantize<short><<<gridU, quantBlock, sharedMemSize>>>(d_U_sqrt, (short*)d_quantized_U_ptr, d_U_scales,
                                                                          m, k, quant_block_rows, quant_block_cols, q_max);
        } else {
            block_quantize<char><<<gridU, quantBlock, sharedMemSize>>>(d_U_sqrt, (char*)d_quantized_U_ptr, d_U_scales,
                                                                         m, k, quant_block_rows, quant_block_cols, q_max);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 对 sqrtVT 量化
        if (quant_type == 16) {
            block_quantize<short><<<gridVT, quantBlock, sharedMemSize>>>(d_sqrtVT, (short*)d_quantized_VT_ptr, d_VT_scales,
                                                                          k, n, quant_block_rows, quant_block_cols, q_max);
        } else {
            block_quantize<char><<<gridVT, quantBlock, sharedMemSize>>>(d_sqrtVT, (char*)d_quantized_VT_ptr, d_VT_scales,
                                                                         k, n, quant_block_rows, quant_block_cols, q_max);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 将量化 scale 从 device 拷贝到 host
        float* h_U_scales = (float*)malloc(sizeof(float)*U_num_blocks);
        CUDA_CHECK(cudaMemcpy(h_U_scales, d_U_scales, sizeof(float)*U_num_blocks, cudaMemcpyDeviceToHost));
        results->U_sqrt_scales = h_U_scales;
        float* h_VT_scales = (float*)malloc(sizeof(float)*VT_num_blocks);
        CUDA_CHECK(cudaMemcpy(h_VT_scales, d_VT_scales, sizeof(float)*VT_num_blocks, cudaMemcpyDeviceToHost));
        results->sqrtVT_scales = h_VT_scales;
        cudaFree(d_U_scales);
        cudaFree(d_VT_scales);
        
        // 将量化结果从 device 拷贝到 host，并释放设备内存
        if (quant_type == 16) {
            short* h_quantized_U = (short*)malloc(sizeof(short)*m*k);
            CUDA_CHECK(cudaMemcpy(h_quantized_U, d_quantized_U_ptr, sizeof(short)*m*k, cudaMemcpyDeviceToHost));
            cudaFree(d_quantized_U_ptr);
            results->quantized_U_sqrt = h_quantized_U;
            
            short* h_quantized_VT = (short*)malloc(sizeof(short)*k*n);
            CUDA_CHECK(cudaMemcpy(h_quantized_VT, d_quantized_VT_ptr, sizeof(short)*k*n, cudaMemcpyDeviceToHost));
            cudaFree(d_quantized_VT_ptr);
            results->quantized_sqrtVT = h_quantized_VT;
        } else {
            char* h_quantized_U = (char*)malloc(sizeof(char)*m*k);
            CUDA_CHECK(cudaMemcpy(h_quantized_U, d_quantized_U_ptr, sizeof(char)*m*k, cudaMemcpyDeviceToHost));
            cudaFree(d_quantized_U_ptr);
            results->quantized_U_sqrt = h_quantized_U;
            
            char* h_quantized_VT = (char*)malloc(sizeof(char)*k*n);
            CUDA_CHECK(cudaMemcpy(h_quantized_VT, d_quantized_VT_ptr, sizeof(char)*k*n, cudaMemcpyDeviceToHost));
            cudaFree(d_quantized_VT_ptr);
            results->quantized_sqrtVT = h_quantized_VT;
        }
    }

    // 15. 释放不再需要的 device 内存
    cudaFree(d_A);
    cudaFree(d_U);
    cudaFree(d_VT);
    cudaFree(d_S);
    cudaFree(d_work);
    cudaFree(devInfo);
    cudaFree(d_S_sqrt);
    cudaFree(d_U_k);
    cudaFree(d_U_sqrt);
    cudaFree(d_VT_k);
    cudaFree(d_sqrtVT);
    cudaFree(d_A_reconstructed);

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUDA_CHECK(cudaDeviceReset());
    free(h_S_sqrt);

    return results;
}

//-----------------------------------------------------------------
// 释放 SVDResults 内存
extern "C" void free_svd_results(SVDResults* res)
{
    if (res) {
        if (res->S) free(res->S);
        if (res->U) free(res->U);
        if (res->VT) free(res->VT);
        if (res->U_sqrt) free(res->U_sqrt);
        if (res->sqrtVT) free(res->sqrtVT);
        if (res->A_reconstructed) free(res->A_reconstructed);
        if (res->U_sqrt_scales) free(res->U_sqrt_scales);
        if (res->sqrtVT_scales) free(res->sqrtVT_scales);
        if (res->quantized_U_sqrt) free(res->quantized_U_sqrt);
        if (res->quantized_sqrtVT) free(res->quantized_sqrtVT);
        free(res);
    }
}

// 模板内核：对量化数据反量化并做矩阵乘法：计算 C = U_deq * VT_deq
// U_deq = (U_q / q_max) * scale_U, VT_deq = (VT_q / q_max) * scale_VT
// U_q 的尺寸为 U_rows x U_cols（即 m x k），VT_q 尺寸为 VT_rows x VT_cols（即 k x n），均采用列主序存储
// quant_block_rows、quant_block_cols 为量化时的块尺寸；
// U_block_rows = ceil(U_rows / quant_block_rows)，VT_block_rows = ceil(VT_rows / quant_block_rows)
// 为了加速，使用共享内存对 U 和 VT 分块加载，并同时加载对应的 scale 数据
template <typename T>
__global__ void dequantizedGemm(const T* U_q, const float* U_scales, int U_rows, int U_cols,
                                int quant_block_rows, int quant_block_cols, int U_block_rows,
                                const T* VT_q, const float* VT_scales, int VT_rows, int VT_cols,
                                int quant_block_rows_vt, int quant_block_cols_vt, int VT_block_rows,
                                float q_max, float* C)
{
    const int TILE_DIM = 16;
    // 分配共享内存：
    // sU、sVT: 存放原始量化数据 tile（类型 T）
    // sU_scales、sVT_scales: 存放对应的 scale（float）
    // sU_deq、sVT_deq: 存放预先反量化后的 tile（float）
    __shared__ T sU[TILE_DIM][TILE_DIM];
    __shared__ T sVT[TILE_DIM][TILE_DIM];
    __shared__ float sU_scales[TILE_DIM][TILE_DIM];
    __shared__ float sVT_scales[TILE_DIM][TILE_DIM];
    __shared__ float sU_deq[TILE_DIM][TILE_DIM];
    __shared__ float sVT_deq[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;  // 输出矩阵 C 的行索引，对应 U_deq 的行
    int col = blockIdx.x * TILE_DIM + threadIdx.x;  // 输出矩阵 C 的列索引，对应 VT_deq 的列
    float sum = 0.0f;

    // 计算内维循环次数：U_cols 与 VT_rows均为 k
    int numTiles = (U_cols + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < numTiles; t++) {
        int p = t * TILE_DIM + threadIdx.x;
        // 加载 U_q 的数据和对应 scale
        if (row < U_rows && p < U_cols) {
            sU[threadIdx.y][threadIdx.x] = U_q[row + p * U_rows];
            int u_block_i = row / quant_block_rows;
            int u_block_j = p / quant_block_cols;
            int u_block_index = u_block_i + u_block_j * U_block_rows;
            sU_scales[threadIdx.y][threadIdx.x] = U_scales[u_block_index];
        } else {
            sU[threadIdx.y][threadIdx.x] = 0;
            sU_scales[threadIdx.y][threadIdx.x] = 0;
        }
        // 加载 VT_q 的数据和对应 scale
        int p2 = t * TILE_DIM + threadIdx.y;
        if (p2 < VT_rows && col < VT_cols) {
            sVT[threadIdx.y][threadIdx.x] = VT_q[p2 + col * VT_rows];
            int vt_block_i = p2 / quant_block_rows_vt;
            int vt_block_j = col / quant_block_cols_vt;
            int vt_block_index = vt_block_i + vt_block_j * VT_block_rows;
            sVT_scales[threadIdx.y][threadIdx.x] = VT_scales[vt_block_index];
        } else {
            sVT[threadIdx.y][threadIdx.x] = 0;
            sVT_scales[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();

        // 预先反量化：在共享内存中计算 tile 内的 U_deq 与 VT_deq
        // 每个线程完成自己位置的反量化计算
        sU_deq[threadIdx.y][threadIdx.x] = ((float)sU[threadIdx.y][threadIdx.x] / q_max) * sU_scales[threadIdx.y][threadIdx.x];
        sVT_deq[threadIdx.y][threadIdx.x] = ((float)sVT[threadIdx.y][threadIdx.x] / q_max) * sVT_scales[threadIdx.y][threadIdx.x];
        __syncthreads();

        // 累加乘积：使用预先反量化后的数据
        for (int j = 0; j < TILE_DIM; j++) {
            sum += sU_deq[threadIdx.y][j] * sVT_deq[j][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < U_rows && col < VT_cols) {
        C[row + col * U_rows] = sum;
    }
}


// 新增接口：在 GPU 上执行反量化与矩阵乘法，返回反量化后 U_deq*VT_deq 结果（host 内存）
extern "C" float* dequantized_matmul_interface(
    void* quantized_U, float* U_scales_host, int U_rows, int U_cols,
    void* quantized_VT, float* VT_scales_host, int VT_rows, int VT_cols,
    int quant_block_rows, int quant_block_cols, int quant_type)
{
    // U_rows = m, U_cols = k, VT_rows = k, VT_cols = n
    int m = U_rows, k = U_cols, n = VT_cols;
    // 计算 U_sqrt 分块时的块数
    int U_block_rows = (m + quant_block_rows - 1) / quant_block_rows;
    int U_blocks_col = (k + quant_block_cols - 1) / quant_block_cols;
    int U_num_blocks = U_block_rows * U_blocks_col;
    // 对 VT：VT_rows = k, VT_cols = n
    int VT_block_rows = (VT_rows + quant_block_rows - 1) / quant_block_rows;  
    int VT_blocks_col = (VT_cols + quant_block_cols - 1) / quant_block_cols;
    int VT_num_blocks = VT_block_rows * VT_blocks_col;

    float q_max = 0.0f;
    if (quant_type == 16) q_max = 32767.0f;
    else if (quant_type == 8) q_max = 127.0f;
    else if (quant_type == 4) q_max = 7.0f;
    else q_max = 1.0f;

    // --- 新增：将 host 上的 scales 拷贝到设备 ---
    float* d_U_scales = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_U_scales, sizeof(float) * U_num_blocks));
    CUDA_CHECK(cudaMemcpy(d_U_scales, U_scales_host, sizeof(float)*U_num_blocks, cudaMemcpyHostToDevice));

    float* d_VT_scales = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_VT_scales, sizeof(float) * VT_num_blocks));
    CUDA_CHECK(cudaMemcpy(d_VT_scales, VT_scales_host, sizeof(float)*VT_num_blocks, cudaMemcpyHostToDevice));
    // --- end ---

    // --- 新增：将 host 上的量化数据拷贝到设备 ---
    void* d_quantized_U = nullptr;
    size_t size_U = (quant_type == 16) ? sizeof(short)*m*k : sizeof(char)*m*k;
    CUDA_CHECK(cudaMalloc(&d_quantized_U, size_U));
    CUDA_CHECK(cudaMemcpy(d_quantized_U, quantized_U, size_U, cudaMemcpyHostToDevice));

    void* d_quantized_VT = nullptr;
    size_t size_VT = (quant_type == 16) ? sizeof(short)*k*n : sizeof(char)*k*n;
    CUDA_CHECK(cudaMalloc(&d_quantized_VT, size_VT));
    CUDA_CHECK(cudaMemcpy(d_quantized_VT, quantized_VT, size_VT, cudaMemcpyHostToDevice));
    // --- end ---

    // 分配设备内存用于输出矩阵 C（m x n，列主序存储）
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeof(float) * m * n));

    const int TILE_DIM = 16;
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((n + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM);
    // 为共享内存分配空间：4个 TILE_DIM*TILE_DIM 的 float 数组
    size_t sharedMemSize = 4 * TILE_DIM * TILE_DIM * sizeof(float);

    // 根据 quant_type 选择内核模板实例
    if (quant_type == 16) {
        dequantizedGemm<short><<<gridDim, blockDim, sharedMemSize>>>(
            (short*)d_quantized_U, d_U_scales, m, k,
            quant_block_rows, quant_block_cols, U_block_rows,
            (short*)d_quantized_VT, d_VT_scales, VT_rows, n,
            quant_block_rows, quant_block_cols, VT_block_rows,
            q_max, d_C);
    } else {
        dequantizedGemm<char><<<gridDim, blockDim, sharedMemSize>>>(
            (char*)d_quantized_U, d_U_scales, m, k,
            quant_block_rows, quant_block_cols, U_block_rows,
            (char*)d_quantized_VT, d_VT_scales, VT_rows, n,
            quant_block_rows, quant_block_cols, VT_block_rows,
            q_max, d_C);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 释放我们在本接口中分配的设备内存用于 scales 和量化数据
    cudaFree(d_U_scales);
    cudaFree(d_VT_scales);
    cudaFree(d_quantized_U);
    cudaFree(d_quantized_VT);

    // 分配 host 内存，并将结果从设备拷贝到 host 后返回
    float* h_C = (float*)malloc(sizeof(float) * m * n);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
    cudaFree(d_C);
    return h_C;
}



#ifdef TEST_MAIN
int main(int argc, char* argv[])
{
    int m = 4096, n = 4096;
    if (argc >= 3) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    }
    float *h_A = (float*)malloc(sizeof(float)*m*n);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            h_A[i+j*m] = static_cast<float>(rand())/RAND_MAX;
        }
    }
    printf("测试 SVD 接口，矩阵尺寸：%d x %d\n", m, n);
    // 例如，传入 INT8 量化
    SVDResults* res = libsvd_quant_multi_interface(h_A, m, n, 8);
    printf("SVD 结果：奇异值个数 k = %d, 量化类型 = %d\n", res->k, res->quant_type);
    free_svd_results(res);
    free(h_A);
    return 0;
}
#endif
