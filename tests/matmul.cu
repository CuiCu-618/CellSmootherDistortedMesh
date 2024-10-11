#include <mma.h>
#include <stdio.h>
using namespace nvcuda;

#define N 8 // Matrix size (8x8)

// Kernel function for matrix multiplication
__global__ void
matMulKernel(double *A, double *B, double *C, int width)
{
  int row = threadIdx.y;
  int col = threadIdx.x;

  {
    double sum = 0.0f;
    for (int k = 0; k < width; ++k)
      {
        sum += A[row * width + k] * B[k * width + col];
      }
    __syncthreads();

    C[row * width + col] = sum;
  }
}


template <int n_dofs_1d, typename Number = double>
__host__ __device__ inline unsigned int
get_base(const unsigned int row, const unsigned int z = 0)
{
  // return 0;

  auto base1 = (row & 3) < 2 ? 0 : 4;
  auto base2 = (z & 1) << 3;
  auto base3 = (z & 3) < 2 ? 0 : 4;

  return base1 ^ base2 ^ base3;
}

template <int n_q_points_1d, bool dof_to_quad, bool add, bool in_place>
__global__ void
matMulKernel1(double *shape_data, double *in, double *out, int width)
{
  const int tid    = (threadIdx.y * 8 + threadIdx.x) & 31;
  const int warpId = threadIdx.y / 4;

  const int row = tid / 4;
  const int col = tid & 3;

  constexpr int offset = 0;

  double2 c[n_q_points_1d / 2];

  int z = 0;
  {
    const int c_idx =
      (row * n_q_points_1d + 2 * col + (z * 2 + warpId) * offset) ^
      get_base<n_q_points_1d>(row, z * 2 + warpId);

    if constexpr (add)
      c[z] = *((double2 *)(out + c_idx));
    else
      c[z] = {0, 0};
  }

  if (add)
    __syncthreads();

  for (int cycle = 0; cycle < 2; ++cycle)
    {
      const int b_idx = dof_to_quad ?
                          ((col + cycle * 4) * n_q_points_1d + row) ^
                            get_base<n_q_points_1d>(col + cycle * 4, 0) :
                          (col + cycle * 4 + n_q_points_1d * row) ^
                            get_base<n_q_points_1d>(row, 0);

      auto b0 = shape_data[b_idx];

      {
        const int a_idx =
          (row * n_q_points_1d + col + cycle * 4 + (z * 2 + warpId) * offset) ^
          get_base<n_q_points_1d>(row, z * 2 + warpId);

        auto a0 = in_place ? out[a_idx] : in[a_idx];

        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                     "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                     : "=d"(c[z].x), "=d"(c[z].y)
                     : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
      }
    }

  if (in_place)
    __syncthreads();

  {
    const int c_idx =
      (row * n_q_points_1d + 2 * col + (z * 2 + warpId) * offset) ^
      get_base<n_q_points_1d>(row, z * 2 + warpId);

    *((double2 *)(out + c_idx)) = c[z];
  }
}

int
main()
{
  int size = N * N * sizeof(double);

  // Allocate memory on host
  double h_A[N][N], h_B[N][N], h_C[N][N] = {};

  // Initialize matrices A and B
  for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < N; j++)
        {
          auto jj    = j ^ get_base<N>(i);
          h_A[i][jj] = static_cast<double>(2 * i + j);
          h_B[i][jj] = static_cast<double>(i - j);
        }
    }

  // Allocate memory on device
  double *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  // Copy matrices A and B to device memory
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Define block and grid dimensions
  dim3 threadsPerBlock(8, 4); // Each block contains 8x8 threads
  dim3 numBlocks(1, 1);       // Only one block is needed for an 8x8 matrix

  // Launch the matrix multiplication kernel
  // matMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

  matMulKernel1<8, 0, 0, 0><<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

  // Check for any errors in kernel launch
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
      return -1;
    }

  // Wait for the kernel to finish
  cudaDeviceSynchronize();

  // Copy the result matrix C back to host memory
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // Print the result matrix
  printf("matrix A:\n");
  for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < N; j++)
        {
          printf("%f ", h_A[i][j]);
        }
      printf("\n");
    }
  printf("matrix B:\n");
  for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < N; j++)
        {
          printf("%f ", h_B[i][j]);
        }
      printf("\n");
    }
  printf("Result matrix C:\n");
  for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < N; j++)
        {
          auto jj = j ^ get_base<N>(i);
          printf("%f ", h_C[i][jj]);
        }
      printf("\n");
    }

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
