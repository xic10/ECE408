#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASKWIDTH 3
#define MASKRADIUS 1
#define TILEWIDTH 8 //Output TileWidth 

//@@ Define constant memory for device kernel here
__constant__ float Cache[MASKWIDTH][MASKWIDTH][MASKWIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int col = blockIdx.x * TILEWIDTH + tx; //blockDim is actually TILEWIDTH + 2 * MASKRADIUS
  int row = blockIdx.y * TILEWIDTH + ty;
  int hgt = blockIdx.z * TILEWIDTH + tz;
  
  int start_x = col - MASKRADIUS;
  int start_y = row - MASKRADIUS;
  int start_z = hgt - MASKRADIUS;
  
  __shared__ float SM[TILEWIDTH + 2 * MASKRADIUS][TILEWIDTH + 2 * MASKRADIUS][TILEWIDTH + 2 * MASKRADIUS];

  //Loading tile into share memory
  if(start_x >= 0 && start_x < x_size && start_y >= 0 && start_y < y_size && start_z >= 0 && start_z < z_size)
    SM[tz][ty][tx] = input[start_z * y_size * x_size + start_y * x_size + start_x];
  else
    SM[tz][ty][tx] = 0; 

  __syncthreads();
  
  float Pvalue = 0;
  if(tx < TILEWIDTH && ty < TILEWIDTH && tz < TILEWIDTH){
    
    for(int i = 0; i < MASKWIDTH; i++) {
      for(int j = 0; j < MASKWIDTH; j++) {
        for(int k = 0; k < MASKWIDTH; k++) {
        Pvalue += Cache[i][j][k] * SM[tz + i][ty + j][tx + k];
        //wbLog(TRACE, "The value is ", Pvalue, "z:" i, "y:", j, "x:" k);
        }
      }
    }
    __syncthreads();
    
    if(col < x_size && row < y_size && hgt < z_size)
      output[hgt * y_size * x_size + row * x_size + col] = Pvalue;
  }
  __syncthreads();

}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  cudaMalloc((void **) &deviceInput, z_size * y_size * x_size * sizeof(float));
  cudaMalloc((void **) &deviceOutput, z_size * y_size * x_size * sizeof(float));

  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  cudaMemcpy(deviceInput, hostInput + 3, z_size * y_size * x_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Cache, hostKernel, MASKWIDTH * MASKWIDTH * MASKWIDTH * sizeof(float));
  
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 DimGrid(ceil(1.0 * x_size / TILEWIDTH), ceil(1.0 * y_size / TILEWIDTH), ceil(1.0 * z_size / TILEWIDTH));
  dim3 DimBlock(TILEWIDTH + 2 * MASKRADIUS, TILEWIDTH + 2 * MASKRADIUS, TILEWIDTH + 2 * MASKRADIUS);
  
  //@@ Launch the GPU kernel here
  
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  cudaMemcpy(hostOutput + 3, deviceOutput, z_size * y_size * x_size * sizeof(float), cudaMemcpyDeviceToHost);
  
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
