#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILEWIDTH 32

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float subTileM[TILEWIDTH][TILEWIDTH];
  __shared__ float subTileN[TILEWIDTH][TILEWIDTH];
  
  int row = blockIdx.y * blockDim.y + threadIdx.y; //The second dimension(y) is numCRows,first dimension(x) is numCColumns
  int col = blockIdx.x * blockDim.x + threadIdx.x; //Kernel will first traverse x(columns) for numColumns, then go to next row
  float curVal = 0.0;
  int tx = threadIdx.x, ty = threadIdx.y;

    for(int i = 0; i < (numAColumns - 1) / TILEWIDTH + 1; i++){ // Deal with numAColumns != TILEWIDTH * N
      if(row < numCRows && i * TILEWIDTH + tx < numAColumns)  //Put 0 in the location of shared memory that outside of the valid range
        subTileM[ty][tx] = A[row * numAColumns + i * TILEWIDTH + tx];
      else
        subTileM[ty][tx] = 0;
      if(col < numCColumns && i * TILEWIDTH + ty < numBRows)
        subTileN[ty][tx] = B[(i * TILEWIDTH + ty) * numBColumns + col];
      else 
         subTileN[ty][tx] = 0;     
      __syncthreads();
      for(int j = 0; j < TILEWIDTH; j++)
        curVal += subTileM[ty][j] * subTileN[j][tx];  
      __syncthreads();
    }
    if(row < numCRows && col < numCColumns)//Must perform this Boundary Check too !!!
      C[row * numCColumns + col] = curVal; //It's actually a one-dimensional array, if don't do the boundary check
  // The situation like numAColumns != TILEWIDTH * N or TILEWIDTH > numAColumns will lead to 0 assignment  
 
    
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **) &deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc((void **) &deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc((void **) &deviceC, numCRows * numCColumns * sizeof(float));
  
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil((1.0 * numCColumns)/TILEWIDTH), ceil((1.0 * numCRows)/TILEWIDTH), 1);
  dim3 DimBlock(TILEWIDTH, TILEWIDTH, 1);
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows* numCColumns* sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}