/**
 * @file evaluate_kernel.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief collection of device functions
 * @version 1.0
 * @date 2022-12-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef CUDA_EVALUATE_CUH
#define CUDA_EVALUATE_CUH

#include <mma.h>

#include "patch_base.cuh"

using namespace nvcuda;


namespace PSMF
{

  template <int dim_m, int dim_n>
  struct Shape
  {
    static constexpr unsigned int m = dim_m;
    static constexpr unsigned int n = dim_n;
  };


  // Function to calculate GCD of two numbers
  __device__ constexpr unsigned int
  gcd(unsigned int a, unsigned int b)
  {
    if (b == 0)
      return a;
    return gcd(b, a % b);
  }

  // Recursive template function to calculate LCM of two numbers
  template <int a, int b>
  struct LCM
  {
    static constexpr unsigned int value = (a * b) / gcd(a, b);
  };

  // Function to calculate the multiple of a number
  template <int n, int constant>
  __device__ constexpr unsigned int
  calculate_multiple()
  {
    // Calculate the multiple of n
    constexpr unsigned int multiple = LCM<n, constant>::value / n;

    return multiple;
  }


  template <int n_dofs_1d, typename Number = double>
  __host__ __device__ inline unsigned int
  get_base(const unsigned int row, const unsigned int z = 0)
  {
    printf("Should never be called!\n");
    return 0;
  }

  template <>
  __host__ __device__ inline unsigned int
  get_base<0, double>(const unsigned int, const unsigned int)
  {
    return 0;
  }

  template <>
  __host__ __device__ inline unsigned int
  get_base<8, double>(const unsigned int row, const unsigned int z)
  {
    auto base1 = (row & 3) < 2 ? 0 : 4;
    auto base2 = (z & 1) << 3;
    auto base3 = (z & 3) < 2 ? 0 : 4;

    return base1 ^ base2 ^ base3;
  }


  ////////////////////////////////////////////////////////////////////
  /////////////////////// TPEvaluatorBase ////////////////////////////
  ////////////////////////////////////////////////////////////////////
  /**
   * A base struct for the various TensorProduct Evaluator template
   * specializations, containing common functionalities.
   *
   * @tparam T Type of the actual vectorized array. We are using the
   *   Couriously Recurring Template Pattern (see
   *   https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern) in
   *   this struct to avoid having to resort to `virtual` member functions.
   */
  template <typename T,
            int n_dofs_1d,
            typename Number,
            LaplaceVariant laplace_type,
            int            dim>
  struct TPEvaluatorBase
  {
    __device__
    TPEvaluatorBase() = default;

    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *laplace_matrix,
          const Number *bilaplace_matrix,
          Number       *tmp)
    {}

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {}
  };

  template <typename T, int n_dofs_1d, typename Number>
  struct TPEvaluatorBase<T, n_dofs_1d, Number, LaplaceVariant::Basic, 2>
  {
    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *laplace_matrix,
          const Number *bilaplace_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, laplace_matrix, bilaplace_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false, bool doubled = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const unsigned int row = threadIdx.y;
      const unsigned int col = threadIdx.x % n_dofs_1d;

      Number pval = 0;
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (unsigned int k = 0; k < n_dofs_1d; ++k)
        {
          const unsigned int shape_idx = row * n_dofs_1d + k;

          const unsigned int source_idx =
            (direction == 0) ? (col * n_dofs_1d + k) : (k * n_dofs_1d + col);

          pval += shape_data[shape_idx] * in[source_idx];
        }


      const unsigned int destination_idx =
        (direction == 0) ? (col * n_dofs_1d + row) : (row * n_dofs_1d + col);

      if (doubled)
        pval *= 2;

      if (add)
        out[destination_idx] += pval;
      else if (sub)
        out[destination_idx] -= pval;
      else
        out[destination_idx] = pval;
    }
  };

  template <typename T, int n_dofs_1d, typename Number>
  struct TPEvaluatorBase<T, n_dofs_1d, Number, LaplaceVariant::Basic, 3>
  {
    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *laplace_matrix,
          const Number *bilaplace_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, laplace_matrix, bilaplace_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false, bool doubled = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int    stride = n_dofs_1d * n_dofs_1d;
      constexpr Number scale  = doubled ? 2 : 1;

      const unsigned int row = threadIdx.y;
      const unsigned int col = threadIdx.x % n_dofs_1d;

      Number pval[n_dofs_1d];
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      for (unsigned int z = 0; z < n_dofs_1d; ++z)
        {
          pval[z] = 0;
          // #pragma unroll
          for (unsigned int k = 0; k < n_dofs_1d; ++k)
            {
              const unsigned int shape_idx = row * n_dofs_1d + k;

              const unsigned int source_idx =
                (direction == 0) ? (col * n_dofs_1d + k + z * stride) :
                (direction == 1) ? (k * n_dofs_1d + col + z * stride) :
                                   (z * n_dofs_1d + col + k * stride);

              pval[z] += shape_data[shape_idx] * in[source_idx];
            }
        }

      for (unsigned int z = 0; z < n_dofs_1d; ++z)
        {
          const unsigned int destination_idx =
            (direction == 0) ? (col * n_dofs_1d + row + z * stride) :
            (direction == 1) ? (row * n_dofs_1d + col + z * stride) :
                               (z * n_dofs_1d + col + row * stride);

          if (add)
            out[destination_idx] += scale * pval[z];
          else if (sub)
            out[destination_idx] -= scale * pval[z];
          else
            out[destination_idx] = scale * pval[z];
        }
    }
  };



  template <typename T, int n_dofs_1d, typename Number>
  struct TPEvaluatorBase<T, n_dofs_1d, Number, LaplaceVariant::ConflictFree, 2>
  {
    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *laplace_matrix,
          const Number *bilaplace_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, laplace_matrix, bilaplace_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false, bool doubled = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const unsigned int row = threadIdx.y;
      const unsigned int col = threadIdx.x % n_dofs_1d;

      Number pval = 0;
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (unsigned int k = 0; k < n_dofs_1d; ++k)
        {
          const unsigned int shape_idx =
            (direction == 0) ? (col * n_dofs_1d + k) : (row * n_dofs_1d + k);

          const unsigned int source_idx =
            (direction == 0) ? (row * n_dofs_1d + k) : (k * n_dofs_1d + col);

          pval += shape_data[shape_idx] * in[source_idx];
        }


      const unsigned int destination_idx = row * n_dofs_1d + col;

      if (doubled)
        pval *= 2;

      if (add)
        out[destination_idx] += pval;
      else if (sub)
        out[destination_idx] -= pval;
      else
        out[destination_idx] = pval;
    }
  };

  template <typename T, int n_dofs_1d, typename Number>
  struct TPEvaluatorBase<T, n_dofs_1d, Number, LaplaceVariant::ConflictFree, 3>
  {
    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *laplace_matrix,
          const Number *bilaplace_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, laplace_matrix, bilaplace_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false, bool doubled = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int    stride = n_dofs_1d * n_dofs_1d;
      constexpr Number scale  = doubled ? 2 : 1;

      const unsigned int row = threadIdx.y;
      const unsigned int col = threadIdx.x % n_dofs_1d;

      Number pval[n_dofs_1d];
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      for (unsigned int z = 0; z < n_dofs_1d; ++z)
        {
          pval[z] = 0;
          // #pragma unroll
          for (unsigned int k = 0; k < n_dofs_1d; ++k)
            {
              const unsigned int shape_idx =
                (direction == 0) ? col * n_dofs_1d + k :
                (direction == 1) ? row * n_dofs_1d + k :
                                   z * n_dofs_1d + k;

              const unsigned int source_idx =
                (direction == 0) ? (row * n_dofs_1d + k + z * stride) :
                (direction == 1) ? (k * n_dofs_1d + col + z * stride) :
                                   (row * n_dofs_1d + col + k * stride);

              pval[z] += shape_data[shape_idx] * in[source_idx];
            }
        }

      for (unsigned int z = 0; z < n_dofs_1d; ++z)
        {
          const unsigned int destination_idx =
            row * n_dofs_1d + col + z * stride;

          if (add)
            out[destination_idx] += scale * pval[z];
          else if (sub)
            out[destination_idx] -= scale * pval[z];
          else
            out[destination_idx] = scale * pval[z];
        }
    }
  };


  template <typename T, int n_dofs_1d>
  struct TPEvaluatorBase<T, n_dofs_1d, double, LaplaceVariant::TensorCore, 3>
  {
    using Number = double;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *laplace_matrix,
          const Number *bilaplace_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, laplace_matrix, bilaplace_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false, bool doubled = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      if (direction == 0)
        {
          const int warpId = (threadIdx.x + threadIdx.y * n_dofs_1d) / 32;

          wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::col_major>
            b_frag;
          wmma::fragment<wmma::accumulator, 8, 8, 4, double>
            c_frag[n_dofs_1d / 2];

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            if (add)
              wmma::load_matrix_sync(c_frag[z],
                                     &out[(z * 2 + warpId) * 8 * 8],
                                     8,
                                     wmma::mem_col_major);
            else
              wmma::fill_fragment(c_frag[z], 0.0f);

          if (add)
            __syncthreads();

          for (int i = 0; i < 2; ++i)
            {
              wmma::load_matrix_sync(a_frag, &shape_data[i * 4], 8);

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  wmma::load_matrix_sync(b_frag,
                                         &in[(z * 2 + warpId) * 8 * 8 + i * 4],
                                         8);

                  wmma::mma_sync(c_frag[z], a_frag, b_frag, c_frag[z]);
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            wmma::store_matrix_sync(&out[(z * 2 + warpId) * 8 * 8],
                                    c_frag[z],
                                    8,
                                    wmma::mem_col_major);
        }

      else if (direction == 1)
        {
          const int warpId = (threadIdx.x + threadIdx.y * n_dofs_1d) / 32;

          wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major>
            b_frag;
          wmma::fragment<wmma::accumulator, 8, 8, 4, double>
            c_frag[n_dofs_1d / 2];

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            if (add)
              wmma::load_matrix_sync(c_frag[z],
                                     &out[(z * 2 + warpId) * 8 * 8],
                                     8,
                                     wmma::mem_row_major);
            else
              wmma::fill_fragment(c_frag[z], 0.0f);

          if (add)
            __syncthreads();

          for (int i = 0; i < 2; ++i)
            {
              wmma::load_matrix_sync(a_frag, &shape_data[i * 4], 8);

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  wmma::load_matrix_sync(
                    b_frag, &in[(z * 2 + warpId) * 8 * 8 + i * 4 * 8], 8);

                  wmma::mma_sync(c_frag[z], a_frag, b_frag, c_frag[z]);
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            wmma::store_matrix_sync(&out[(z * 2 + warpId) * 8 * 8],
                                    c_frag[z],
                                    8,
                                    wmma::mem_row_major);
        }
      else
        {
          constexpr int multiple = std::is_same<Number, double>::value ?
                                     calculate_multiple<n_dofs_1d, 16>() :
                                     calculate_multiple<n_dofs_1d, 32>();

          constexpr int stride = n_dofs_1d * n_dofs_1d;

          const int row = threadIdx.y;
          const int col = threadIdx.x % n_dofs_1d;

          Number pval[n_dofs_1d];
          // kernel product: A kdot src, [N x N] * [N^dim, 1]
          for (int z = 0; z < n_dofs_1d; ++z)
            {
              pval[z] = 0;
              // #pragma unroll
              for (int k = 0; k < n_dofs_1d; ++k)
                {
                  const int shape_idx =
                    (direction == 0) ?
                      col * n_dofs_1d + (k + col / multiple) % n_dofs_1d :
                    (direction == 1) ? row * n_dofs_1d + k :
                                       z * n_dofs_1d + k;

                  const int source_idx =
                    (direction == 0) ?
                      (row * n_dofs_1d + (k + col / multiple) % n_dofs_1d +
                       z * stride) :
                    (direction == 1) ? (k * n_dofs_1d + col + z * stride) :
                                       (row * n_dofs_1d + col + k * stride);

                  pval[z] += shape_data[shape_idx] * in[source_idx];
                }
            }

          for (int z = 0; z < n_dofs_1d; ++z)
            {
              const int destination_idx = row * n_dofs_1d + col + z * stride;

              if (add)
                out[destination_idx] += pval[z];
              else if (sub)
                out[destination_idx] -= pval[z];
              else
                out[destination_idx] = pval[z];
            }
        }
    }
  };



  template <typename T, int n_dofs_1d>
  struct TPEvaluatorBase<T, n_dofs_1d, double, LaplaceVariant::TensorCoreMMA, 3>
  {
    using Number = double;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *laplace_matrix,
          const Number *bilaplace_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, laplace_matrix, bilaplace_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false, bool doubled = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      if (direction == 0)
        {
          const int tid    = (threadIdx.y * 8 + threadIdx.x) & 31;
          const int warpId = threadIdx.y / 4;

          const int row = tid / 4;
          const int col = tid & 3;

          constexpr int offset = n_dofs_1d * n_dofs_1d;

          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 2 + warpId) * offset) ^
                get_base<0>(row, z * 2 + warpId);

              if constexpr (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          if (add)
            __syncthreads();

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int b_idx =
                (col + cycle * 4 + n_dofs_1d * row) ^ get_base<0>(row, 0);

              auto b0 = shape_data[b_idx];

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int a_idx = (row * n_dofs_1d + col + cycle * 4 +
                                     (z * 2 + warpId) * offset) ^
                                    get_base<0>(row, z * 2 + warpId);

                  auto a0 = in[a_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 2 + warpId) * offset) ^
                get_base<0>(row, z * 2 + warpId);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
      else if (direction == 1)
        {
          const int tid    = (threadIdx.y * 8 + threadIdx.x) & 31;
          const int warpId = threadIdx.y / 4;

          const int row = tid / 4;
          const int col = tid & 3;

          constexpr int offset = n_dofs_1d * n_dofs_1d;

          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 2 + warpId) * offset) ^
                get_base<0>(row, z * 2 + warpId);

              if constexpr (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          if (add)
            __syncthreads();

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx =
                (row * n_dofs_1d + col + cycle * 4) ^ get_base<0>(row, 0);
              auto a0 = shape_data[a_idx];

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int b_idx =
                    ((col + cycle * 4) * n_dofs_1d + row +
                     (z * 2 + warpId) * offset) ^
                    get_base<0>(col + cycle * 4, z * 2 + warpId);

                  auto b0 = in[b_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 2 + warpId) * offset) ^
                get_base<0>(row, z * 2 + warpId);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
      else
        {
          const int tid    = (threadIdx.y * 8 + threadIdx.x) & 31;
          const int warpId = threadIdx.y / 4;

          const int row = tid / 4;
          const int col = tid & 3;

          constexpr int offset = n_dofs_1d * n_dofs_1d;

          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((z * 2 + warpId) * n_dofs_1d + 2 * col + row * offset) ^
                get_base<0>(z * 2 + warpId, row);

              if constexpr (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          if (add)
            __syncthreads();

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx =
                (row * n_dofs_1d + col + cycle * 4) ^ get_base<0>(row, 0);
              auto a0 = shape_data[a_idx];

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int b_idx =
                    ((z * 2 + warpId) * n_dofs_1d + row +
                     (col + cycle * 4) * offset) ^
                    get_base<0>(z * 2 + warpId, col + cycle * 4);

                  auto b0 = in[b_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((z * 2 + warpId) * n_dofs_1d + 2 * col + row * offset) ^
                get_base<0>(z * 2 + warpId, row);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
    }
  };



  template <typename T, int n_dofs_1d>
  struct TPEvaluatorBase<T,
                         n_dofs_1d,
                         double,
                         LaplaceVariant::TensorCoreMMACF,
                         3>
  {
    using Number = double;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *laplace_matrix,
          const Number *bilaplace_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, laplace_matrix, bilaplace_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false, bool doubled = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      if (direction == 0)
        {
          const int tid    = (threadIdx.y * 8 + threadIdx.x) & 31;
          const int warpId = threadIdx.y / 4;

          const int row = tid / 4;
          const int col = tid & 3;

          constexpr int offset = n_dofs_1d * n_dofs_1d;

          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 2 + warpId) * offset) ^
                get_base<n_dofs_1d>(row, z * 2 + warpId);

              if constexpr (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          if (add)
            __syncthreads();

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int b_idx = (col + cycle * 4 + n_dofs_1d * row) ^
                                get_base<n_dofs_1d>(row, 0);

              auto b0 = shape_data[b_idx];

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int a_idx = (row * n_dofs_1d + col + cycle * 4 +
                                     (z * 2 + warpId) * offset) ^
                                    get_base<n_dofs_1d>(row, z * 2 + warpId);

                  auto a0 = in[a_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 2 + warpId) * offset) ^
                get_base<n_dofs_1d>(row, z * 2 + warpId);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
      else if (direction == 1)
        {
          const int tid    = (threadIdx.y * 8 + threadIdx.x) & 31;
          const int warpId = threadIdx.y / 4;

          const int row = tid / 4;
          const int col = tid & 3;

          constexpr int offset = n_dofs_1d * n_dofs_1d;

          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 2 + warpId) * offset) ^
                get_base<n_dofs_1d>(row, z * 2 + warpId);

              if constexpr (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          if (add)
            __syncthreads();

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx = (row * n_dofs_1d + col + cycle * 4) ^
                                get_base<n_dofs_1d>(row, 0);
              auto a0 = shape_data[a_idx];

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int b_idx =
                    ((col + cycle * 4) * n_dofs_1d + row +
                     (z * 2 + warpId) * offset) ^
                    get_base<n_dofs_1d>(col + cycle * 4, z * 2 + warpId);

                  auto b0 = in[b_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 2 + warpId) * offset) ^
                get_base<n_dofs_1d>(row, z * 2 + warpId);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
      else
        {
          const int tid    = (threadIdx.y * 8 + threadIdx.x) & 31;
          const int warpId = threadIdx.y / 4;

          const int row = tid / 4;
          const int col = tid & 3;

          constexpr int offset = n_dofs_1d * n_dofs_1d;

          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((z * 2 + warpId) * n_dofs_1d + 2 * col + row * offset) ^
                get_base<n_dofs_1d>(z * 2 + warpId, row);

              if constexpr (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          if (add)
            __syncthreads();

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx = (row * n_dofs_1d + col + cycle * 4) ^
                                get_base<n_dofs_1d>(row, 0);
              auto a0 = shape_data[a_idx];

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int b_idx =
                    ((z * 2 + warpId) * n_dofs_1d + row +
                     (col + cycle * 4) * offset) ^
                    get_base<n_dofs_1d>(z * 2 + warpId, col + cycle * 4);

                  auto b0 = in[b_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((z * 2 + warpId) * n_dofs_1d + 2 * col + row * offset) ^
                get_base<n_dofs_1d>(z * 2 + warpId, row);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
    }
  };



  ////////////////////////////////////////////////////////////////////
  /////////////////// TPEvaluatorBiLaplace ///////////////////////////
  ////////////////////////////////////////////////////////////////////
  template <LaplaceVariant laplace_type,
            typename Number,
            int n_dofs_1d,
            int dim>
  struct TPEvaluatorBilaplace
    : TPEvaluatorBase<
        TPEvaluatorBilaplace<laplace_type, Number, n_dofs_1d, dim>,
        n_dofs_1d,
        Number,
        laplace_type,
        dim>
  {
    using TPEvaluatorBase<
      TPEvaluatorBilaplace<laplace_type, Number, n_dofs_1d, dim>,
      n_dofs_1d,
      Number,
      laplace_type,
      dim>::apply;
    __device__ void
    vmult_impl(Number       *dst,
               const Number *src,
               const Number *mass_matrix,
               const Number *laplace_matrix,
               const Number *bilaplace_matrix,
               Number       *tmp)
    {}
  };

  template <LaplaceVariant laplace_type, typename Number, int n_dofs_1d>
  struct TPEvaluatorBilaplace<laplace_type, Number, n_dofs_1d, 2>
    : TPEvaluatorBase<TPEvaluatorBilaplace<laplace_type, Number, n_dofs_1d, 2>,
                      n_dofs_1d,
                      Number,
                      laplace_type,
                      2>
  {
    using TPEvaluatorBase<
      TPEvaluatorBilaplace<laplace_type, Number, n_dofs_1d, 2>,
      n_dofs_1d,
      Number,
      laplace_type,
      2>::apply;

    __device__ void
    vmult_impl(Number       *dst,
               const Number *src,
               const Number *mass_matrix,
               const Number *laplace_matrix,
               const Number *bilaplace_matrix,
               Number       *tmp)
    {
      constexpr unsigned int offset = n_dofs_1d * n_dofs_1d;

      apply<0, false>(mass_matrix, src, tmp);
      __syncthreads();
      apply<1, true>(&bilaplace_matrix[offset], tmp, dst);
      __syncthreads();

      apply<0, false>(bilaplace_matrix, src, tmp);
      __syncthreads();
      apply<1, true>(&mass_matrix[offset], tmp, dst);
      __syncthreads();

      apply<0, false>(laplace_matrix, src, tmp);
      __syncthreads();
      apply<1, true, false, true>(&laplace_matrix[offset], tmp, dst);
    }
  };

  template <LaplaceVariant laplace_type, typename Number, int n_dofs_1d>
  struct TPEvaluatorBilaplace<laplace_type, Number, n_dofs_1d, 3>
    : TPEvaluatorBase<TPEvaluatorBilaplace<laplace_type, Number, n_dofs_1d, 3>,
                      n_dofs_1d,
                      Number,
                      laplace_type,
                      3>
  {
    using TPEvaluatorBase<
      TPEvaluatorBilaplace<laplace_type, Number, n_dofs_1d, 3>,
      n_dofs_1d,
      Number,
      laplace_type,
      3>::apply;

    __device__ void
    vmult_impl(Number       *dst,
               const Number *src,
               const Number *mass_matrix,
               const Number *laplace_matrix,
               const Number *bilaplace_matrix,
               Number       *tmp)
    {
      constexpr unsigned int local_dim = Util::pow(n_dofs_1d, 3);
      constexpr unsigned int offset    = n_dofs_1d * n_dofs_1d;

      // BxMxM + MxBxM + MxMxB
      apply<0, false>(mass_matrix, src, &tmp[local_dim]);
      __syncthreads();
      apply<1, false>(&mass_matrix[offset], &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, false>(&bilaplace_matrix[offset * 2], tmp, dst);
      __syncthreads();
      apply<1, false>(&bilaplace_matrix[offset], &tmp[local_dim], tmp);
      __syncthreads();
      apply<0, false>(bilaplace_matrix, src, &tmp[local_dim]);
      __syncthreads();
      apply<1, true>(&mass_matrix[offset], &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, true>(&mass_matrix[offset * 2], tmp, dst);
      __syncthreads();

      // 2(LxLxM + LxMxL + MxLxL)
      apply<0, false>(laplace_matrix, src, &tmp[local_dim]);
      __syncthreads();
      apply<1, false>(&laplace_matrix[offset], &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, true, false, true>(&mass_matrix[offset * 2], tmp, dst);
      __syncthreads();
      apply<1, false>(&mass_matrix[offset], &tmp[local_dim], tmp);
      __syncthreads();
      apply<0, false>(mass_matrix, src, &tmp[local_dim]);
      __syncthreads();
      apply<1, true>(&laplace_matrix[offset], &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, true, false, true>(&laplace_matrix[offset * 2], tmp, dst);
    }
  };


  ////////////////////////////////////////////////////////////////////
  //////////////////// TPEvaluatorSmoother ///////////////////////////
  ////////////////////////////////////////////////////////////////////
  template <typename Number,
            int            n_dofs_1d,
            LaplaceVariant laplace_type,
            int            dim>
  struct TPEvaluatorSmootherVmult
    : TPEvaluatorBase<
        TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace_type, dim>,
        n_dofs_1d,
        Number,
        laplace_type,
        dim>
  {
    using TPEvaluatorBase<
      TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace_type, dim>,
      n_dofs_1d,
      Number,
      laplace_type,
      dim>::apply;

    __device__ void
    vmult_impl(Number       *dst,
               const Number *src,
               const Number *mass_matrix,
               const Number *laplace_matrix,
               const Number *bilaplace_matrix,
               Number       *tmp)
    {}
  };


  template <typename Number, int n_dofs_1d, LaplaceVariant laplace_type>
  struct TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace_type, 2>
    : TPEvaluatorBase<
        TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace_type, 2>,
        n_dofs_1d,
        Number,
        laplace_type,
        2>
  {
    using TPEvaluatorBase<
      TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace_type, 2>,
      n_dofs_1d,
      Number,
      laplace_type,
      2>::apply;

    __device__ void
    vmult_impl(Number       *dst,
               const Number *src,
               const Number *mass_matrix,
               const Number *laplace_matrix,
               const Number *bilaplace_matrix,
               Number       *tmp)
    {
      constexpr unsigned int offset = n_dofs_1d * n_dofs_1d;

      apply<0, false>(mass_matrix, src, tmp);
      __syncthreads();
      apply<1, false, true>(&bilaplace_matrix[offset], tmp, dst);
      __syncthreads();
      apply<0, false>(bilaplace_matrix, src, tmp);
      __syncthreads();
      apply<1, false, true>(mass_matrix, tmp, dst);
      __syncthreads();
      apply<0, false>(laplace_matrix, src, tmp);
      __syncthreads();
      apply<1, false, true, true>(laplace_matrix, tmp, dst);
    }
  };


  template <typename Number, int n_dofs_1d, LaplaceVariant laplace_type>
  struct TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace_type, 3>
    : TPEvaluatorBase<
        TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace_type, 3>,
        n_dofs_1d,
        Number,
        laplace_type,
        3>
  {
    using TPEvaluatorBase<
      TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace_type, 3>,
      n_dofs_1d,
      Number,
      laplace_type,
      3>::apply;

    __device__ void
    vmult_impl(Number       *dst,
               const Number *src,
               const Number *mass_matrix,
               const Number *laplace_matrix,
               const Number *bilaplace_matrix,
               Number       *tmp)
    {
      constexpr unsigned int local_dim = Util::pow(n_dofs_1d, 3);
      constexpr unsigned int offset    = n_dofs_1d * n_dofs_1d;

      // BxMxM + MxBxM + MxMxB
      apply<0, false>(mass_matrix, src, &tmp[local_dim]);
      __syncthreads();
      apply<1, false>(mass_matrix, &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, false, true>(&bilaplace_matrix[offset * 2], tmp, dst);
      __syncthreads();
      apply<1, false>(&bilaplace_matrix[offset], &tmp[local_dim], tmp);
      __syncthreads();
      apply<0, false>(bilaplace_matrix, src, &tmp[local_dim]);
      __syncthreads();
      apply<1, true>(mass_matrix, &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, false, true>(mass_matrix, tmp, dst);
      __syncthreads();

      // 2(LxLxM + LxMxL + MxLxL)
      apply<0, false>(laplace_matrix, src, &tmp[local_dim]);
      __syncthreads();
      apply<1, false>(laplace_matrix, &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, false, true, true>(mass_matrix, tmp, dst);
      __syncthreads();
      apply<1, false>(mass_matrix, &tmp[local_dim], tmp);
      __syncthreads();
      apply<0, false>(mass_matrix, src, &tmp[local_dim]);
      __syncthreads();
      apply<1, true>(laplace_matrix, &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, false, true, true>(laplace_matrix, tmp, dst);
    }
  };


  template <typename Number, int n_dofs_1d, SmootherVariant smoother, int dim>
  struct TPEvaluatorSmootherInv
  {
    __device__ void
    apply_inverse(Number       *dst,
                  Number       *src,
                  const Number *eigenvalues,
                  const Number *eigenvectors,
                  Number       *tmp)
    {}

    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {}
  };

  // Bila, KSVD
  template <typename Number, int n_dofs_1d>
  struct TPEvaluatorSmootherInv<Number, n_dofs_1d, SmootherVariant::GLOBAL, 2>
  {
    __device__ void
    apply_inverse(Number       *dst,
                  Number       *src,
                  const Number *eigenvalues,
                  const Number *eigenvectors,
                  Number       *tmp)
    {
      constexpr unsigned int offset = n_dofs_1d * n_dofs_1d;

      apply<0, true>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, true>(&eigenvectors[offset], tmp, src);
      __syncthreads();
      src[threadIdx.y * n_dofs_1d + threadIdx.x % n_dofs_1d] /=
        (0 + eigenvalues[n_dofs_1d + threadIdx.y] +
         eigenvalues[threadIdx.x % n_dofs_1d]);
      __syncthreads();
      apply<0, false>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, false, true>(&eigenvectors[offset], tmp, dst);
    }

    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const unsigned int linear_tid =
        threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;
      const unsigned int row = linear_tid / n_dofs_1d;
      const unsigned int col = linear_tid % n_dofs_1d;

      Number pval = 0;

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (unsigned int k = 0; k < n_dofs_1d; ++k)
        {
          const unsigned int shape_idx =
            contract_over_rows ?
              ((direction == 0) ? k * n_dofs_1d + col : k * n_dofs_1d + row) :
              ((direction == 0) ? col * n_dofs_1d + k : row * n_dofs_1d + k);

          const unsigned int source_idx =
            (direction == 0) ? (row * n_dofs_1d + k) : (k * n_dofs_1d + col);

          pval += shape_data[shape_idx] * in[source_idx];
        }

      {
        const unsigned int destination_idx = row * n_dofs_1d + col;

        if (add)
          out[destination_idx] += pval;
        else
          out[destination_idx] = pval;
      }
    }
  };

  template <typename Number, int n_dofs_1d>
  struct TPEvaluatorSmootherInv<Number, n_dofs_1d, SmootherVariant::GLOBAL, 3>
  {
    __device__ void
    apply_inverse(Number       *dst,
                  Number       *src,
                  const Number *eigenvalues,
                  const Number *eigenvectors,
                  Number       *tmp)
    {
      constexpr unsigned int local_dim = Util::pow(n_dofs_1d, 3);

      constexpr unsigned int offset = n_dofs_1d * n_dofs_1d;

      apply<0, true>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, true>(&eigenvectors[offset], tmp, &tmp[local_dim]);
      __syncthreads();
      apply<2, true>(&eigenvectors[offset * 2], &tmp[local_dim], tmp);
      __syncthreads();
      for (unsigned int z = 0; z < n_dofs_1d; ++z)
        {
          tmp[z * n_dofs_1d * n_dofs_1d + threadIdx.y * n_dofs_1d +
              threadIdx.x % n_dofs_1d] /=
            (0 + eigenvalues[z + 2 * n_dofs_1d] +
             eigenvalues[threadIdx.y + n_dofs_1d] +
             eigenvalues[threadIdx.x % n_dofs_1d]);
        }
      __syncthreads();
      apply<0, false>(eigenvectors, tmp, &tmp[local_dim]);
      __syncthreads();
      apply<1, false>(&eigenvectors[offset], &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, false, true>(&eigenvectors[offset * 2], tmp, dst);
    }

    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr unsigned int stride = n_dofs_1d * n_dofs_1d;

      const unsigned int linear_tid =
        threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;
      const unsigned int row = linear_tid / n_dofs_1d;
      const unsigned int col = linear_tid % n_dofs_1d;

      Number pval[n_dofs_1d];

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      for (unsigned int z = 0; z < n_dofs_1d; ++z)
        {
          pval[z] = 0;
          // #pragma unroll
          for (unsigned int k = 0; k < n_dofs_1d; ++k)
            {
              const unsigned int shape_idx =
                contract_over_rows ? k * n_dofs_1d + row : row * n_dofs_1d + k;

              const unsigned int source_idx =
                (direction == 0) ? (col * n_dofs_1d + k + z * stride) :
                (direction == 1) ? (k * n_dofs_1d + col + z * stride) :
                                   (z * n_dofs_1d + col + k * stride);

              pval[z] += shape_data[shape_idx] * in[source_idx];
            }
        }

      for (unsigned int z = 0; z < n_dofs_1d; ++z)
        {
          const unsigned int destination_idx =
            (direction == 0) ? (col * n_dofs_1d + row + z * stride) :
            (direction == 1) ? (row * n_dofs_1d + col + z * stride) :
                               (z * n_dofs_1d + col + row * stride);
          if (add)
            out[destination_idx] += pval[z];
          else
            out[destination_idx] = pval[z];
        }
    }
  };


  // Bila, KSVD
  template <typename Number, int n_dofs_1d>
  struct TPEvaluatorSmootherInv<Number,
                                n_dofs_1d,
                                SmootherVariant::ConflictFree,
                                2>
  {
    __device__ void
    apply_inverse(Number       *dst,
                  Number       *src,
                  const Number *eigenvalues,
                  const Number *eigenvectors,
                  Number       *tmp)
    {
      constexpr unsigned int n_dofs_1d_i = n_dofs_1d - 2;
      constexpr unsigned int offset      = n_dofs_1d_i * n_dofs_1d_i;

      const unsigned int linear_tid =
        threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

      unsigned int row = linear_tid / n_dofs_1d_i;
      unsigned int col = linear_tid % n_dofs_1d_i;

      apply<0, true>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, true>(&eigenvectors[offset], tmp, src);
      __syncthreads();
      if (linear_tid < n_dofs_1d_i * n_dofs_1d_i)
        src[row * n_dofs_1d_i + col] /=
          (0 + eigenvalues[n_dofs_1d_i + row] + eigenvalues[col]);
      __syncthreads();
      apply<0, false>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, false, true>(&eigenvectors[offset], tmp, dst);
    }

    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr unsigned int n_dofs_1d_i = n_dofs_1d - 2;
      const unsigned int     linear_tid =
        threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

      const unsigned int row = linear_tid / n_dofs_1d_i;
      const unsigned int col = linear_tid % n_dofs_1d_i;

      Number pval = 0;

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      if (linear_tid < n_dofs_1d_i * n_dofs_1d_i)
        {
          // #pragma unroll
          for (unsigned int k = 0; k < n_dofs_1d_i; ++k)
            {
              const unsigned int shape_idx =
                contract_over_rows ?
                  ((direction == 0) ? k * n_dofs_1d_i + col :
                                      k * n_dofs_1d_i + row) :
                  ((direction == 0) ? col * n_dofs_1d_i + k :
                                      row * n_dofs_1d_i + k);

              const unsigned int source_idx = (direction == 0) ?
                                                (row * n_dofs_1d_i + k) :
                                                (k * n_dofs_1d_i + col);

              pval += shape_data[shape_idx] * in[source_idx];
            }
        }

      if (linear_tid < n_dofs_1d_i * n_dofs_1d_i)
        {
          const unsigned int destination_idx = row * n_dofs_1d_i + col;

          if (add)
            out[destination_idx] += pval;
          else
            out[destination_idx] = pval;
        }
    }
  };

  template <typename Number, int n_dofs_1d>
  struct TPEvaluatorSmootherInv<Number,
                                n_dofs_1d,
                                SmootherVariant::ConflictFree,
                                3>
  {
    __device__ void
    apply_inverse(Number       *dst,
                  Number       *src,
                  const Number *eigenvalues,
                  const Number *eigenvectors,
                  Number       *tmp)
    {
      const unsigned int linear_tid =
        threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

      constexpr unsigned int n_dofs_1d_i = n_dofs_1d - 2;
      constexpr unsigned int offset      = n_dofs_1d_i * n_dofs_1d_i;

      unsigned int row = linear_tid / n_dofs_1d_i;
      unsigned int col = linear_tid % n_dofs_1d_i;

      apply<0, true>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, true>(&eigenvectors[offset], tmp, src);
      __syncthreads();
      apply<2, true>(&eigenvectors[offset * 2], src, tmp);
      __syncthreads();
      if (linear_tid < n_dofs_1d_i * n_dofs_1d_i)
        for (unsigned int z = 0; z < n_dofs_1d - 2; ++z)
          {
            tmp[z * n_dofs_1d * n_dofs_1d + row * n_dofs_1d_i + col] /=
              (eigenvalues[z + 2 * n_dofs_1d_i] +
               eigenvalues[row + n_dofs_1d_i] + eigenvalues[col]);
          }
      __syncthreads();
      apply<0, false>(eigenvectors, tmp, src);
      __syncthreads();
      apply<1, false>(&eigenvectors[offset], src, tmp);
      __syncthreads();
      apply<2, false, true>(&eigenvectors[offset * 2], tmp, dst);
    }

    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr unsigned int stride      = n_dofs_1d * n_dofs_1d;
      constexpr unsigned int n_dofs_1d_i = n_dofs_1d - 2;
      const unsigned int     linear_tid =
        threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

      const unsigned int row = linear_tid / n_dofs_1d_i;
      const unsigned int col = linear_tid % n_dofs_1d_i;

      Number pval[n_dofs_1d_i];

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      if (linear_tid < n_dofs_1d_i * n_dofs_1d_i)
        for (unsigned int z = 0; z < n_dofs_1d_i; ++z)
          {
            pval[z] = 0;
            // #pragma unroll
            for (unsigned int k = 0; k < n_dofs_1d_i; ++k)
              {
                const unsigned int shape_idx =
                  contract_over_rows ?
                    ((direction == 0) ? k * n_dofs_1d_i + col :
                     (direction == 1) ? k * n_dofs_1d_i + row :
                                        k * n_dofs_1d_i + z) :
                    ((direction == 0) ? col * n_dofs_1d_i + k :
                     (direction == 1) ? row * n_dofs_1d_i + k :
                                        z * n_dofs_1d_i + k);

                const unsigned int source_idx =
                  (direction == 0) ? (row * n_dofs_1d_i + k + z * stride) :
                  (direction == 1) ? (k * n_dofs_1d_i + col + z * stride) :
                                     (row * n_dofs_1d_i + col + k * stride);

                pval[z] += shape_data[shape_idx] * in[source_idx];
              }
          }

      if (linear_tid < n_dofs_1d_i * n_dofs_1d_i)
        for (unsigned int z = 0; z < n_dofs_1d_i; ++z)
          {
            const unsigned int destination_idx =
              row * n_dofs_1d_i + col + z * stride;

            if (add)
              out[destination_idx] += pval[z];
            else
              out[destination_idx] = pval[z];
          }
    }
  };


  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////

  template <int dim, int fe_degree, typename Number, LaplaceVariant laplace>
  __device__ void
  evaluate_laplace(const unsigned int         local_patch,
                   SharedDataOp<dim, Number> *shared_data)
  {
#ifdef TENSORCORE
    constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;
#else
    constexpr unsigned int n_dofs_1d = 2 * fe_degree + 1;
#endif
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);

    TPEvaluatorBilaplace<laplace, Number, n_dofs_1d, dim> eval;
    __syncthreads();

    eval.vmult(
      &shared_data->local_dst[local_patch * local_dim],
      &shared_data->local_src[local_patch * local_dim],
      &shared_data->local_mass[local_patch * n_dofs_1d * n_dofs_1d * dim],
      &shared_data->local_laplace[local_patch * n_dofs_1d * n_dofs_1d * dim],
      &shared_data->local_bilaplace[local_patch * n_dofs_1d * n_dofs_1d * dim],
      &shared_data->tmp[local_patch * local_dim * (dim - 1)]);
    __syncthreads();
  }


  template <int dim,
            int fe_degree,
            typename Number,
            LaplaceVariant     laplace,
            LocalSolverVariant solver>
  __device__ void
  evaluate_smooth_cf(
    const unsigned int local_patch,
    const unsigned int patch,
    SharedDataSmoother<dim, Number, SmootherVariant::ConflictFree, solver>
                                                                  *shared_data,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data *gpu_data)
  {
    constexpr unsigned int n_dofs_1d   = 2 * fe_degree + 1;
    constexpr unsigned int n_dofs_1d_z = dim == 2 ? 1 : n_dofs_1d - 2;
    constexpr unsigned int local_dim   = Util::pow(n_dofs_1d, dim);

    TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace, dim> eval_vmult;
    TPEvaluatorSmootherInv<Number,
                           n_dofs_1d,
                           SmootherVariant::ConflictFree,
                           dim>
      eval_inverse;
    __syncthreads();

    eval_vmult.vmult(
      &shared_data->local_src[local_patch * local_dim],
      &shared_data->local_dst[local_patch * local_dim],
      &shared_data->local_mass[local_patch * n_dofs_1d * n_dofs_1d],
      &shared_data->local_laplace[local_patch * n_dofs_1d * n_dofs_1d],
      &shared_data->local_bilaplace[local_patch * n_dofs_1d * n_dofs_1d * dim],
      &shared_data->tmp[local_patch * local_dim * 2]);
    __syncthreads();

    const unsigned int linear_tid =
      threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

    if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
      {
        unsigned int row = linear_tid / (n_dofs_1d - 2) + 1;
        unsigned int col = linear_tid % (n_dofs_1d - 2) + 1;

        for (unsigned int z = 0; z < n_dofs_1d_z; ++z)
          {
            shared_data
              ->tmp[2 * local_patch * local_dim + z * n_dofs_1d * n_dofs_1d +
                    (row - 1) * (n_dofs_1d - 2) + col - 1] =
              shared_data->local_dst[local_patch * local_dim +
                                     (z + dim - 2) * n_dofs_1d * n_dofs_1d +
                                     row * n_dofs_1d + col];

            shared_data->tmp[2 * local_patch * local_dim + local_dim +
                             z * n_dofs_1d * n_dofs_1d +
                             (row - 1) * (n_dofs_1d - 2) + col - 1] =
              shared_data->local_src[local_patch * local_dim +
                                     (z + dim - 2) * n_dofs_1d * n_dofs_1d +
                                     row * n_dofs_1d + col];
          }
      }

    if (solver == LocalSolverVariant::Exact)
      {
        __syncthreads();
        return;
      }

    if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
      {
        unsigned int row = linear_tid / (n_dofs_1d - 2);
        unsigned int col = linear_tid % (n_dofs_1d - 2);

        unsigned int patch_type = 0;
        for (unsigned int d = 0; d < dim; ++d)
          patch_type += gpu_data->patch_type[patch * dim + d] * Util::pow(3, d);

        for (unsigned int d = 0; d < dim; ++d)
          {
            shared_data->local_mass[(local_patch + d) * (n_dofs_1d - 2) + col] =
              gpu_data
                ->eigenvalues[(patch_type * dim + d) * (n_dofs_1d - 2) + col];

            shared_data
              ->local_bilaplace[((local_patch * dim + d) * (n_dofs_1d - 2) +
                                 row) *
                                  (n_dofs_1d - 2) +
                                col] =
              gpu_data->eigenvectors[((patch_type * dim + d) * (n_dofs_1d - 2) +
                                      row) *
                                       (n_dofs_1d - 2) +
                                     col];
          }
      }
    __syncthreads();

    eval_inverse.apply_inverse(
      &shared_data->tmp[local_patch * local_dim * 2],
      &shared_data->tmp[local_patch * local_dim * 2 + local_dim],
      &shared_data->local_mass[local_patch * n_dofs_1d * n_dofs_1d],
      &shared_data->local_bilaplace[local_patch * n_dofs_1d * n_dofs_1d * dim],
      &shared_data->local_src[local_patch * local_dim]);
    __syncthreads();
  }


  template <int dim, int fe_degree, typename Number, LocalSolverVariant solver>
  __device__ void
  evaluate_smooth_global(
    const unsigned int local_patch,
    SharedDataSmoother<dim, Number, SmootherVariant::GLOBAL, solver>
      *shared_data)
  {
    constexpr unsigned int n_dofs_1d = 2 * fe_degree - 1;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);

    TPEvaluatorSmootherInv<Number, n_dofs_1d, SmootherVariant::GLOBAL, dim>
      eval;
    __syncthreads();

    eval.apply_inverse(
      &shared_data->local_dst[local_patch * local_dim],
      &shared_data->local_src[local_patch * local_dim],
      &shared_data->local_mass[local_patch * n_dofs_1d * dim],
      &shared_data->local_laplace[local_patch * n_dofs_1d * n_dofs_1d * dim],
      &shared_data->tmp[local_patch * local_dim * (dim - 1)]);
    __syncthreads();
  }

} // namespace PSMF


#endif // CUDA_EVALUATE_CUH
