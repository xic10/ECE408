
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

# define TILE_WIDTH 16

namespace mxnet
{
namespace op
{
// __constant__ float filter[24*12*5*5];
__global__ void shared_memory_convolution(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */
    // y: output  B x M x H_out x W_out
    // x: input   B x C x H x W
    // k: filter  M x C x K x K
    //const int B = x.shape_[0]; batch size
    //const int M = y.shape_[1]; output channel
    //const int C = x.shape_[1]; input channel
    //const int H = x.shape_[2]; input height
    //const int W = x.shape_[3]; input width
    //const int K = k.shape_[3]; filter size
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) filter[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
    int n,m,h0,w0,h_base,w_base,h,w;
    int X_tile_width = TILE_WIDTH + K - 1;
    extern __shared__ float shmem[];
    float* X_shared = &shmem[0];
    n = blockIdx.x;
    m = blockIdx.y;
    h0 = threadIdx.x;
    w0 = threadIdx.y;
    h_base = (blockIdx.z/W_grid)*TILE_WIDTH;
    w_base = (blockIdx.z%W_grid)*TILE_WIDTH;
    h = h_base+h0;
    w = w_base+w0;
    float acc=0;
    for (int c=0;c<C;c++){
        for(int i = h;i<h_base+X_tile_width;i+=TILE_WIDTH){
            for(int j = w;j<w_base+X_tile_width;j+=TILE_WIDTH){
                if (i<H && j<W)
                    X_shared[(i-h_base)*X_tile_width + j-w_base]=x4d(n,c,i,j);
                else
                    X_shared[(i-h_base)*X_tile_width + j-w_base]=0;
            }
        }
        
        __syncthreads();
        for(int p=0;p<K;p++){
            for(int q=0;q<K;q++){
                acc+=X_shared[(h0+p)*X_tile_width+w0+q]*k4d(m,c,p,q);
            }
        }
        __syncthreads();
    }
    if (h < H_out && w < W_out)
        y4d(n,m,h,w) = acc;
    
#undef y4d
#undef x4d
#undef k4d
}

__global__ void unroll(int b, float* X_unroll, int size, float* x, int C, int K, int H, int W) {
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    int t = blockDim.x*blockIdx.x + threadIdx.x;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    if (t < C * K * K * H_out * W_out){
        int row = t / (H_out*W_out);
        int col = t % (H_out*W_out);
        int q = row % K;
        row /= K;
        int p = row % K;
        int c = row / K;
        int w = col % W_out;
        int h = col / W_out;
        X_unroll[t] = x4d(b,c,h+p,w+q);
    }
#undef x4d
}

__global__ void shared_memory_matrix_multiply(int b, float *A, float *B, float *C,
                                              int numARows, int numAColumns,
                                              int numBRows, int numBColumns,
                                              int numCRows, int numCColumns) {
    __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float CValue = 0;
    
    for (int m = 0; m < (numAColumns - 1) / TILE_WIDTH + 1; ++m){
        if(Row < numARows && m * TILE_WIDTH + tx < numAColumns)
        subTileA[ty][tx] = A[Row * numAColumns + m * TILE_WIDTH + tx];
        else
        subTileA[ty][tx] = 0;
        if(m * TILE_WIDTH + ty < numBRows && Col < numBColumns)
        subTileB[ty][tx] = B[(m * TILE_WIDTH + ty) * numBColumns + Col];
        else
        subTileB[ty][tx] = 0;
        
        __syncthreads();
        
        if(Row < numCRows && Col < numCColumns)
        for(int k = 0; k < TILE_WIDTH; ++k)
            CValue += subTileA[ty][k] * subTileB[k][tx];
        
        __syncthreads();
    }
    
    if(Row < numCRows && Col < numCColumns)
        C[b * numCRows * numCColumns + Row * numCColumns + Col] = CValue;
}


/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
 
    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0]; //batch size
    const int M = y.shape_[1]; //output channel
    const int C = x.shape_[1]; //input channel
    const int H = x.shape_[2]; //input height
    const int W = x.shape_[3]; //input width
    const int K = w.shape_[3]; //filter size
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
    const int H_grid = ceil(1.0 * H_out / TILE_WIDTH);
    const int Z = H_grid * W_grid;

    
    // __constant__ float filter[24*12*5*5];
    // Weight matrix (kernel values) in constant memory
    // cudaMemcpyToSymbol(filter,w.dptr_,M*C*K*K*sizeof(float));
    

    // Unroll
    float* X_unroll;
    cudaMalloc((void **) &X_unroll,C*K*K*H_out*W_out*sizeof(float));
    int num_threads = C * K * K * H_out * W_out;
    int num_blocks = ceil(num_threads / 1024);
    // Matrix Multiplication
    dim3 gridDim(ceil(H_out * W_out/16.0), ceil(M/16.0), 1);
    dim3 blockDim(16.0, 16.0, 1);
    for(int b = 0; b < B;b++){
        unroll<<<num_blocks, 1024>>>(b, X_unroll, num_threads, x.dptr_, C, K, H, W);
        cudaDeviceSynchronize();
        shared_memory_matrix_multiply<<<gridDim, blockDim>>>(b, w.dptr_, X_unroll, y.dptr_, M, C*K*K, C*K*K, H_out*W_out, M, H_out*W_out);
        cudaDeviceSynchronize();
    }
    
    // Shared Memory convolution
    // dim3 gridDim(B, M, Z);
    // dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);
    // size_t shmem_size = sizeof(float) * ( (TILE_WIDTH + K-1)*(TILE_WIDTH + K-1) );
    // shared_memory_convolution<<<gridDim, blockDim, shmem_size>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif