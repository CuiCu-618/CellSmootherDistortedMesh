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
          const Number *derivative_matrix,
          Number       *tmp)
    {}

    template <typename shapeA,
              typename shapeB,
              bool add,
              int  dir,
              int  g_row,
              int  g_col,
              int  cycle>
    __device__ void
    mma_op(const Number *shape_data, const Number *in, Number *out)
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
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
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
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int stride = n_dofs_1d * n_dofs_1d;

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
            out[destination_idx] += pval[z];
          else if (sub)
            out[destination_idx] -= pval[z];
          else
            out[destination_idx] = pval[z];
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
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int multiple = Util::calculate_multiple<n_dofs_1d, 16>();

      const unsigned int row = threadIdx.y;
      const unsigned int col = threadIdx.x % n_dofs_1d;

      Number pval = 0;
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (unsigned int k = 0; k < n_dofs_1d; ++k)
        {
          const unsigned int shape_idx =
            (direction == 0) ?
              (col * n_dofs_1d + (k + col / multiple) % n_dofs_1d) :
              (row * n_dofs_1d + k);

          const unsigned int source_idx =
            (direction == 0) ?
              (row * n_dofs_1d + (k + col / multiple) % n_dofs_1d) :
              (k * n_dofs_1d + col);

          pval += shape_data[shape_idx] * in[source_idx];
        }


      const unsigned int destination_idx = row * n_dofs_1d + col;

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
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int multiple = Util::calculate_multiple<n_dofs_1d, 16>();
      constexpr int stride   = n_dofs_1d * n_dofs_1d;

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
                (direction == 0) ?
                  col * n_dofs_1d + (k + col / multiple) % n_dofs_1d :
                (direction == 1) ? row * n_dofs_1d + k :
                                   z * n_dofs_1d + k;

              const unsigned int source_idx =
                (direction == 0) ?
                  (row * n_dofs_1d + (k + col / multiple) % n_dofs_1d +
                   z * stride) :
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
            out[destination_idx] += pval[z];
          else if (sub)
            out[destination_idx] -= pval[z];
          else
            out[destination_idx] = pval[z];
        }
    }
  };


  template <typename T>
  struct TPEvaluatorBase<T, 8, double, LaplaceVariant::TensorCoreMMA, 2>
  {
    using Number = double;
    using s8x4   = Shape<8, 4>;

    static constexpr unsigned int n_dofs_1d = 8;

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
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <typename shapeA,
              typename shapeB,
              bool add,
              int  dir,
              int  g_row,
              int  g_col,
              int  cycle>
    __device__ void
    mma_op(const Number *shape_data, const Number *in, Number *out)
    {
      const unsigned int tid = (threadIdx.y * 8 + threadIdx.x);

      const unsigned int row = tid / 4;
      const unsigned int col = tid % 4;

      // const unsigned int a_idx =
      //   (row + g_row * 8) * n_dofs_1d + col + cycle * 4;
      // const unsigned int b_idx =
      //   (dir == 0) ? (row + g_col * 8) * n_dofs_1d + col + cycle * 4 :
      //                (col + cycle * 4) * n_dofs_1d + row + g_col * 8;
      // const unsigned int cd_idx =
      //   (dir == 0) ? (2 * col + g_col * 8) * n_dofs_1d + row + g_row * 8 :
      //                (row + g_row * 8) * n_dofs_1d + 2 * col + g_col * 8;

      const unsigned int a_idx = (row)*n_dofs_1d + col + cycle * 4;
      const unsigned int b_idx = (dir == 0) ?
                                   (row)*n_dofs_1d + col + cycle * 4 :
                                   (col + cycle * 4) * n_dofs_1d + row;
      const unsigned int cd_idx =
        (dir == 0) ? (2 * col) * n_dofs_1d + row : (row)*n_dofs_1d + 2 * col;

      // const bool is_avtive_a = row < shapeA::m && col < shapeA::n;
      // const bool is_avtive_b = row < shapeB::m && col < shapeB::n;

      // const bool is_avtive_cd = row < shapeA::m && col < shapeB::m / 2;

      // Number tmp = 0;

      constexpr unsigned int stride = (dir == 0) ? n_dofs_1d : 1;

      // auto &d0 = is_avtive_cd ? out[cd_idx] : tmp;
      // auto &d1 = is_avtive_cd ? out[cd_idx + stride] : tmp;

      // auto a0 = is_avtive_a ? shape_data[a_idx] : 0;
      // auto b0 = is_avtive_b ? in[b_idx] : 0;

      // auto c0 = (is_avtive_cd && add) ? out[cd_idx] : 0;
      // auto c1 = (is_avtive_cd && add) ? out[cd_idx + stride] : 0;

      auto &d0 = out[cd_idx];
      auto &d1 = out[cd_idx + stride];

      // auto a0 = shape_data[a_idx];
      // auto b0 = in[b_idx];

      auto c0 = (add) ? out[cd_idx] : 0;
      auto c1 = (add) ? out[cd_idx + stride] : 0;

      if (tid > 31)
        return;

      asm("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=d"(d0), "=d"(d1)
          : "d"(shape_data[a_idx]), "d"(in[b_idx]), "d"(c0), "d"(c1));
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      // constexpr unsigned int skew_double = Util::padding;

      // mma_op<s8x4, s8x4, add, direction, 0, 0, 0>(shape_data, in, out);
      // __syncthreads();
      // mma_op<s8x4, s8x4, true, direction, 0, 0, 1>(shape_data, in, out);

      const unsigned int tid = (threadIdx.y * 8 + threadIdx.x);

      const unsigned int row = tid / 4;
      const unsigned int col = tid % 4;

      constexpr unsigned int stride = (direction == 0) ? n_dofs_1d : 1;

      for (unsigned int cycle = 0; cycle < 2; ++cycle)
        {
          const unsigned int a_idx  = (row)*n_dofs_1d + col + cycle * 4;
          const unsigned int b_idx  = (direction == 0) ?
                                        (row)*n_dofs_1d + col + cycle * 4 :
                                        (col + cycle * 4) * n_dofs_1d + row;
          const unsigned int cd_idx = (direction == 0) ?
                                        (2 * col) * n_dofs_1d + row :
                                        (row)*n_dofs_1d + 2 * col;

          auto &d0 = out[cd_idx];
          auto &d1 = out[cd_idx + stride];

          auto a0 = shape_data[a_idx];
          auto b0 = in[b_idx];

          auto c0 = (add || cycle == 1) ? out[cd_idx] : 0;
          auto c1 = (add || cycle == 1) ? out[cd_idx + stride] : 0;

          if (tid > 31)
            return;

          asm("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
              "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
              : "=d"(d0), "=d"(d1)
              : "d"(a0), "d"(b0), "d"(c0), "d"(c1));
        }
    }
  };



  template <typename T>
  struct TPEvaluatorBase<T, 8, double, LaplaceVariant::TensorCoreMMA, 3>
  {
    using Number = double;
    using s8x4   = Shape<8, 4>;

    static constexpr unsigned int n_dofs_1d = 8;

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
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr unsigned int skew_double       = Util::padding;
      constexpr unsigned int n_dofs_1d_padding = n_dofs_1d + skew_double;

      const unsigned int tid    = (threadIdx.y * 8 + threadIdx.x) % 32;
      const unsigned int warpId = (threadIdx.y * 8 + threadIdx.x) / 32;

      const unsigned int row = tid / 4;
      const unsigned int col = tid % 4;

      constexpr unsigned int stride = 1;
      constexpr unsigned int offset = n_dofs_1d * n_dofs_1d_padding;

      if (direction == 0)
        {
          for (unsigned int cycle = 0; cycle < 2; ++cycle)
            {
              const unsigned int b_idx =
                (col + cycle * 4) * n_dofs_1d_padding + row ^
                (Util::get_permute_base<3>(col, 0));
              auto b0 = shape_data[b_idx];

              for (unsigned int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const unsigned int a_idx =
                    row * n_dofs_1d_padding + (col + cycle * 4) ^
                    (Util::get_permute_base<3>(row, z * 2 + warpId)) +
                      (z * 2 + warpId) * offset;
                  const unsigned int cd_idx =
                    row * n_dofs_1d_padding + (2 * col) ^
                    (Util::get_permute_base<3>(row, z * 2 + warpId)) +
                      (z * 2 + warpId) * offset;

                  auto  a0 = in[a_idx];
                  auto &d0 = out[cd_idx];
                  auto &d1 = out[cd_idx + stride];
                  auto  c0 = (cycle == 1) ? out[cd_idx] : 0;
                  auto  c1 = (cycle == 1) ? out[cd_idx + stride] : 0;

                  asm("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                      "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                      : "=d"(d0), "=d"(d1)
                      : "d"(a0), "d"(b0), "d"(c0), "d"(c1));
                }
            }
        }
      else if (direction == 1)
        {
          for (unsigned int cycle = 0; cycle < 2; ++cycle)
            {
              const unsigned int a_idx =
                row * n_dofs_1d_padding + (col + cycle * 4) ^
                (Util::get_permute_base<3>(row, 0));
              auto a0 = shape_data[a_idx];

              for (unsigned int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const unsigned int b_idx =
                    (col + cycle * 4) * n_dofs_1d_padding + row ^
                    (Util::get_permute_base<3>(col, z * 2 + warpId)) +
                      (z * 2 + warpId) * offset;
                  const unsigned int cd_idx =
                    row * n_dofs_1d_padding + (2 * col) ^
                    (Util::get_permute_base<3>(row, z * 2 + warpId)) +
                      (z * 2 + warpId) * offset;

                  auto  b0 = in[b_idx];
                  auto &d0 = out[cd_idx];
                  auto &d1 = out[cd_idx + stride];
                  auto  c0 = (add || cycle == 1) ? out[cd_idx] : 0;
                  auto  c1 = (add || cycle == 1) ? out[cd_idx + stride] : 0;

                  asm("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                      "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                      : "=d"(d0), "=d"(d1)
                      : "d"(a0), "d"(b0), "d"(c0), "d"(c1));
                }
            }
        }
      else if (direction == 2)
        {
          for (unsigned int cycle = 0; cycle < 2; ++cycle)
            {
              const unsigned int a_idx =
                row * n_dofs_1d_padding + (col + cycle * 4) ^
                (Util::get_permute_base<3>(row, 0));
              auto a0 = sub ? -shape_data[a_idx] : shape_data[a_idx];

              for (unsigned int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const unsigned int b_idx =
                    (z * 2 + warpId) * n_dofs_1d_padding + row ^
                    (Util::get_permute_base<3>(z * 2 + warpId, col)) +
                      (col + cycle * 4) * offset;
                  const unsigned int cd_idx =
                    (z * 2 + warpId) * n_dofs_1d_padding + (2 * col) ^
                    (Util::get_permute_base<3>(z * 2 + warpId, row)) +
                      row * offset;

                  auto  b0 = in[b_idx];
                  auto &d0 = out[cd_idx];
                  auto &d1 = out[cd_idx + stride];
                  auto  c0 = (add || sub || cycle == 1) ? out[cd_idx] : 0;
                  auto  c1 =
                    (add || sub || cycle == 1) ? out[cd_idx + stride] : 0;

                  asm("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                      "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                      : "=d"(d0), "=d"(d1)
                      : "d"(a0), "d"(b0), "d"(c0), "d"(c1));
                }
            }
        }
    }
  };



  template <typename T>
  struct TPEvaluatorBase<T, 8, double, LaplaceVariant::TensorCore, 2>
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
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr unsigned int n_dofs_1d   = 8;
      constexpr unsigned int skew_double = Util::padding;

      if (direction == 0)
        {
          const unsigned int warpId =
            (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;

          wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::col_major>
                                                             b_frag;
          wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;

          wmma::fill_fragment(c_frag, 0.0f);

          if (warpId != 0)
            return;

          for (int i = 0; i < 2; ++i)
            {
              wmma::load_matrix_sync(b_frag,
                                     &shape_data[i * 4],
                                     8 + skew_double);

              wmma::load_matrix_sync(
                a_frag,
                &in[warpId * 8 * (8 + skew_double) + i * 4],
                8 + skew_double);

              wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

          wmma::store_matrix_sync(&out[warpId * 8 * (8 + skew_double)],
                                  c_frag,
                                  8 + skew_double,
                                  wmma::mem_row_major);
        }
      else if (direction == 1)
        {
          const unsigned int warpId =
            (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;

          wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major>
                                                             b_frag;
          wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;

          if (add)
            wmma::load_matrix_sync(c_frag,
                                   &out[warpId * 8 * (8 + skew_double)],
                                   8 + skew_double,
                                   wmma::mem_row_major);
          else
            wmma::fill_fragment(c_frag, 0.0f);

          if (add)
            __syncthreads();

          if (warpId != 0)
            return;

          for (int i = 0; i < 2; ++i)
            {
              wmma::load_matrix_sync(a_frag,
                                     &shape_data[i * 4],
                                     8 + skew_double);

              wmma::load_matrix_sync(
                b_frag,
                &in[warpId * 8 * (8 + skew_double) + i * 4 * (8 + skew_double)],
                8 + skew_double);

              wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

          wmma::store_matrix_sync(&out[warpId * 8 * (8 + skew_double)],
                                  c_frag,
                                  8 + skew_double,
                                  wmma::mem_row_major);
        }
    }
  };

  template <typename T>
  struct TPEvaluatorBase<T, 8, double, LaplaceVariant::TensorCore, 3>
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
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr unsigned int n_dofs_1d   = 8;
      constexpr unsigned int skew_double = Util::padding;

      if (direction == 0)
        {
          const unsigned int warpId =
            (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;
          wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::col_major>
            b_frag;
          wmma::fragment<wmma::accumulator, 8, 8, 4, double>
            c_frag[n_dofs_1d / 2];

          for (unsigned int z = 0; z < n_dofs_1d / 2; ++z)
            wmma::fill_fragment(c_frag[z], 0.0f);

          for (int i = 0; i < 2; ++i)
            {
              wmma::load_matrix_sync(b_frag,
                                     &shape_data[i * 4],
                                     8 + skew_double);

              for (unsigned int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  wmma::load_matrix_sync(
                    a_frag,
                    &in[(z * 2 + warpId) * 8 * (8 + skew_double) + i * 4],
                    8 + skew_double);

                  wmma::mma_sync(c_frag[z], a_frag, b_frag, c_frag[z]);
                }
            }

          for (unsigned int z = 0; z < n_dofs_1d / 2; ++z)
            wmma::store_matrix_sync(
              &out[(z * 2 + warpId) * 8 * (8 + skew_double)],
              c_frag[z],
              8 + skew_double,
              wmma::mem_row_major);
        }
      else if (direction == 1)
        {
          const unsigned int warpId =
            (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;

          wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major>
            b_frag;
          wmma::fragment<wmma::accumulator, 8, 8, 4, double>
            c_frag[n_dofs_1d / 2];

          for (unsigned int z = 0; z < n_dofs_1d / 2; ++z)
            if (add)
              wmma::load_matrix_sync(
                c_frag[z],
                &out[(z * 2 + warpId) * 8 * (8 + skew_double)],
                8 + skew_double,
                wmma::mem_row_major);
            else
              wmma::fill_fragment(c_frag[z], 0.0f);

          if (add)
            __syncthreads();

          for (int i = 0; i < 2; ++i)
            {
              wmma::load_matrix_sync(a_frag,
                                     &shape_data[i * 4],
                                     8 + skew_double);

              for (unsigned int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  wmma::load_matrix_sync(
                    b_frag,
                    &in[(z * 2 + warpId) * 8 * (8 + skew_double) +
                        i * 4 * (8 + skew_double)],
                    8 + skew_double);

                  wmma::mma_sync(c_frag[z], a_frag, b_frag, c_frag[z]);
                }
            }

          for (unsigned int z = 0; z < n_dofs_1d / 2; ++z)
            wmma::store_matrix_sync(
              &out[(z * 2 + warpId) * 8 * (8 + skew_double)],
              c_frag[z],
              8 + skew_double,
              wmma::mem_row_major);
        }
      else if (direction == 2)
        {
          constexpr int n_dofs_1d_padding = n_dofs_1d + skew_double;
          constexpr int stride            = n_dofs_1d * n_dofs_1d_padding;

          const int row = threadIdx.y;
          const int col = threadIdx.x % n_dofs_1d;

          double pval[n_dofs_1d];

          // kernel product: A kdot src, [N x N] * [N^dim, 1]
          for (unsigned int z = 0; z < n_dofs_1d; ++z)
            {
              pval[z] = 0;
              // #pragma unroll
              for (int k = 0; k < n_dofs_1d; ++k)
                {
                  const unsigned int shape_idx = z * n_dofs_1d_padding + k;
                  const unsigned int source_idx =
                    row * n_dofs_1d_padding + col + k * stride;

                  pval[z] += shape_data[shape_idx] * in[source_idx];
                }
            }

          for (unsigned int z = 0; z < n_dofs_1d; ++z)
            {
              const unsigned int destination_idx =
                row * n_dofs_1d_padding + col + z * stride;

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

  template <typename T>
  struct TPEvaluatorBase<T, 16, double, LaplaceVariant::TensorCore, 2>
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
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr unsigned int n_dofs_1d   = 16;
      constexpr unsigned int skew_double = Util::padding;

      if (direction == 0)
        {
          const unsigned int warpId =
            (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;
          const unsigned int subId = warpId % 4;

          wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::col_major>
                                                             b_frag;
          wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;

          wmma::fill_fragment(c_frag, 0.0f);

          if (warpId > 3)
            return;

          for (int i = 0; i < 4; ++i)
            {
              wmma::load_matrix_sync(
                b_frag,
                &shape_data[subId / 2 * (16 + skew_double) * 8 + i * 4],
                16 + skew_double);

              wmma::load_matrix_sync(
                a_frag,
                &in[subId % 2 * (16 + skew_double) * 8 + i * 4],
                16 + skew_double);

              wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

          wmma::store_matrix_sync(
            &out[subId % 2 * (16 + skew_double) * 8 + subId / 2 * 8],
            c_frag,
            16 + skew_double,
            wmma::mem_row_major);
        }
      else if (direction == 1)
        {
          const unsigned int warpId =
            (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;
          const unsigned int subId = warpId % 4;

          wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major>
                                                             b_frag;
          wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;

          if (add)
            wmma::load_matrix_sync(
              c_frag,
              &out[subId / 2 * (16 + skew_double) * 8 + subId % 2 * 8],
              16 + skew_double,
              wmma::mem_row_major);
          else
            wmma::fill_fragment(c_frag, 0.0f);

          if (add)
            __syncthreads();

          if (warpId > 3)
            return;

          for (int i = 0; i < 4; ++i)
            {
              wmma::load_matrix_sync(
                a_frag,
                &shape_data[subId / 2 * (16 + skew_double) * 8 + i * 4],
                16 + skew_double);

              wmma::load_matrix_sync(
                b_frag,
                &in[subId % 2 * 8 + i * 4 * (16 + skew_double)],
                16 + skew_double);

              wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

          wmma::store_matrix_sync(
            &out[subId / 2 * (16 + skew_double) * 8 + subId % 2 * 8],
            c_frag,
            16 + skew_double,
            wmma::mem_row_major);
        }
    }
  };

  template <typename T>
  struct TPEvaluatorBase<T, 16, double, LaplaceVariant::TensorCore, 3>
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
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr unsigned int n_dofs_1d   = 16;
      constexpr unsigned int skew_double = Util::padding;

      if (direction == 0)
        {
          const unsigned int warpId =
            (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;
          // sub matrix id
          const unsigned int subId = warpId % 4;

          wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::col_major>
            b_frag;
          wmma::fragment<wmma::accumulator, 8, 8, 4, double>
            c_frag[n_dofs_1d / 2];

          for (unsigned int z = 0; z < n_dofs_1d / 2; ++z)
            wmma::fill_fragment(c_frag[z], 0.0f);

          for (int i = 0; i < 4; ++i)
            {
              wmma::load_matrix_sync(
                b_frag,
                &shape_data[subId / 2 * (16 + skew_double) * 8 + i * 4],
                16 + skew_double);

              for (unsigned int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  wmma::load_matrix_sync(
                    a_frag,
                    &in[(z * 2 + warpId / 4) * (16 + skew_double) * 16 +
                        subId % 2 * (16 + skew_double) * 8 + i * 4],
                    16 + skew_double);

                  wmma::mma_sync(c_frag[z], a_frag, b_frag, c_frag[z]);
                }
            }

          for (unsigned int z = 0; z < n_dofs_1d / 2; ++z)
            wmma::store_matrix_sync(
              &out[(z * 2 + warpId / 4) * (16 + skew_double) * 16 +
                   subId % 2 * (16 + skew_double) * 8 + subId / 2 * 8],
              c_frag[z],
              16 + skew_double,
              wmma::mem_row_major);
        }
      else if (direction == 1)
        {
          const unsigned int warpId =
            (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;
          // sub matrix id
          const unsigned int subId = warpId % 4;

          wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major>
            b_frag;
          wmma::fragment<wmma::accumulator, 8, 8, 4, double>
            c_frag[n_dofs_1d / 2];

          for (unsigned int z = 0; z < n_dofs_1d / 2; ++z)
            if (add)
              wmma::load_matrix_sync(
                c_frag[z],
                &out[(z * 2 + warpId / 4) * (16 + skew_double) * 16 +
                     subId / 2 * (16 + skew_double) * 8 + subId % 2 * 8],
                16 + skew_double,
                wmma::mem_row_major);
            else
              wmma::fill_fragment(c_frag[z], 0.0f);

          if (add)
            __syncthreads();

          for (int i = 0; i < 4; ++i)
            {
              wmma::load_matrix_sync(
                a_frag,
                &shape_data[subId / 2 * (16 + skew_double) * 8 + i * 4],
                16 + skew_double);

              for (unsigned int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  wmma::load_matrix_sync(
                    b_frag,
                    &in[(z * 2 + warpId / 4) * (16 + skew_double) * 16 +
                        subId % 2 * 8 + i * 4 * (16 + skew_double)],
                    16 + skew_double);

                  wmma::mma_sync(c_frag[z], a_frag, b_frag, c_frag[z]);
                }
            }

          for (unsigned int z = 0; z < n_dofs_1d / 2; ++z)
            wmma::store_matrix_sync(
              &out[(z * 2 + warpId / 4) * (16 + skew_double) * 16 +
                   subId / 2 * (16 + skew_double) * 8 + subId % 2 * 8],
              c_frag[z],
              16,
              wmma::mem_row_major);
        }
      else
        {
          constexpr int n_dofs_1d_padding = n_dofs_1d + skew_double;
          constexpr int stride            = n_dofs_1d * n_dofs_1d_padding;

          const int row = threadIdx.y;
          const int col = threadIdx.x % n_dofs_1d;

          double pval[n_dofs_1d];

          // kernel product: A kdot src, [N x N] * [N^dim, 1]
          for (unsigned int z = 0; z < n_dofs_1d; ++z)
            {
              pval[z] = 0;
              // #pragma unroll
              for (int k = 0; k < n_dofs_1d; ++k)
                {
                  const unsigned int shape_idx = z * n_dofs_1d_padding + k;
                  const unsigned int source_idx =
                    row * n_dofs_1d_padding + col + k * stride;

                  pval[z] += shape_data[shape_idx] * in[source_idx];
                }
            }

          for (unsigned int z = 0; z < n_dofs_1d; ++z)
            {
              const unsigned int destination_idx =
                row * n_dofs_1d_padding + col + z * stride;

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


  template <typename T>
  struct TPEvaluatorBase<T, 16, float, LaplaceVariant::TensorCore, 2>
  {
    using Number = float;
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
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr unsigned int n_dofs_1d   = 16;
      constexpr unsigned int skew_double = Util::padding;

      if (direction == 0)
        {
          const unsigned int warpId =
            (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;

          wmma::fragment<wmma::matrix_a,
                         16,
                         16,
                         8,
                         wmma::precision::tf32,
                         wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b,
                         16,
                         16,
                         8,
                         wmma::precision::tf32,
                         wmma::col_major>
                                                              b_frag;
          wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;

          wmma::fill_fragment(c_frag, 0.0f);

          if (warpId != 0)
            return;

          for (int i = 0; i < 2; ++i)
            {
              wmma::load_matrix_sync(b_frag,
                                     &shape_data[i * 8],
                                     16 + skew_double);
              // #pragma unroll
              //               for (int t = 0; t < a_frag.num_elements; t++)
              //                 {
              //                   a_frag.x[t] =
              //                   wmma::__float_to_tf32(a_frag.x[t]);
              //                 }

              wmma::load_matrix_sync(
                a_frag,
                &in[warpId * 16 * (16 + skew_double) + i * 8],
                16 + skew_double);
              // #pragma unroll
              //               for (int t = 0; t < b_frag.num_elements; t++)
              //                 {
              //                   b_frag.x[t] =
              //                   wmma::__float_to_tf32(b_frag.x[t]);
              //                 }

              wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

          wmma::store_matrix_sync(&out[warpId * 16 * (16 + skew_double)],
                                  c_frag,
                                  16 + skew_double,
                                  wmma::mem_row_major);
        }
      else if (direction == 1)
        {
          const unsigned int warpId =
            (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;

          wmma::fragment<wmma::matrix_a,
                         16,
                         16,
                         8,
                         wmma::precision::tf32,
                         wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b,
                         16,
                         16,
                         8,
                         wmma::precision::tf32,
                         wmma::row_major>
                                                              b_frag;
          wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;

          if (add)
            wmma::load_matrix_sync(c_frag,
                                   &out[warpId * 16 * (16 + skew_double)],
                                   16 + skew_double,
                                   wmma::mem_row_major);
          else
            wmma::fill_fragment(c_frag, 0.0f);

          if (warpId != 0)
            return;

          if (add)
            __syncthreads();

          for (int i = 0; i < 2; ++i)
            {
              wmma::load_matrix_sync(a_frag,
                                     &shape_data[i * 8],
                                     16 + skew_double);
#pragma unroll
              for (int t = 0; t < a_frag.num_elements; t++)
                {
                  a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
                }

              wmma::load_matrix_sync(b_frag,
                                     &in[warpId * 16 * (16 + skew_double) +
                                         i * 8 * (16 + skew_double)],
                                     16 + skew_double);
#pragma unroll
              for (int t = 0; t < b_frag.num_elements; t++)
                {
                  b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
                }
              wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

          wmma::store_matrix_sync(&out[warpId * 16 * (16 + skew_double)],
                                  c_frag,
                                  16 + skew_double,
                                  wmma::mem_row_major);
        }
    }
  };

  template <typename T>
  struct TPEvaluatorBase<T, 16, float, LaplaceVariant::TensorCore, 3>
  {
    using Number = float;
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
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr unsigned int n_dofs_1d   = 16;
      constexpr unsigned int skew_double = Util::padding;

      if (direction == 0)
        {
          const unsigned int warpId =
            (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;

          wmma::fragment<wmma::matrix_a,
                         16,
                         16,
                         8,
                         wmma::precision::tf32,
                         wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b,
                         16,
                         16,
                         8,
                         wmma::precision::tf32,
                         wmma::col_major>
            b_frag;
          wmma::fragment<wmma::accumulator, 16, 16, 8, float>
            c_frag[n_dofs_1d / 8];

          for (unsigned int z = 0; z < n_dofs_1d / 8; ++z)
            wmma::fill_fragment(c_frag[z], 0.0f);

          for (int i = 0; i < 2; ++i)
            {
              wmma::load_matrix_sync(b_frag,
                                     &shape_data[i * 8],
                                     16 + skew_double);
              // #pragma unroll
              //               for (int t = 0; t < a_frag.num_elements; t++)
              //                 {
              //                   a_frag.x[t] =
              //                   wmma::__float_to_tf32(a_frag.x[t]);
              //                 }
              for (unsigned int z = 0; z < n_dofs_1d / 8; ++z)
                {
                  wmma::load_matrix_sync(
                    a_frag,
                    &in[(z * 8 + warpId) * 16 * (16 + skew_double) + i * 8],
                    16 + skew_double);
                  // #pragma unroll
                  //                   for (int t = 0; t < b_frag.num_elements;
                  //                   t++)
                  //                     {
                  //                       b_frag.x[t] =
                  //                       wmma::__float_to_tf32(b_frag.x[t]);
                  //                     }

                  wmma::mma_sync(c_frag[z], a_frag, b_frag, c_frag[z]);
                }
            }

          for (unsigned int z = 0; z < n_dofs_1d / 8; ++z)
            wmma::store_matrix_sync(
              &out[(z * 8 + warpId) * 16 * (16 + skew_double)],
              c_frag[z],
              16 + skew_double,
              wmma::mem_row_major);
        }
      else if (direction == 1)
        {
          const unsigned int warpId =
            (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;

          wmma::fragment<wmma::matrix_a,
                         16,
                         16,
                         8,
                         wmma::precision::tf32,
                         wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b,
                         16,
                         16,
                         8,
                         wmma::precision::tf32,
                         wmma::row_major>
            b_frag;
          wmma::fragment<wmma::accumulator, 16, 16, 8, float>
            c_frag[n_dofs_1d / 8];

          for (unsigned int z = 0; z < n_dofs_1d / 8; ++z)
            if (add)
              wmma::load_matrix_sync(
                c_frag[z],
                &out[(z * 8 + warpId) * 16 * (16 + skew_double)],
                16 + skew_double,
                wmma::mem_row_major);
            else
              wmma::fill_fragment(c_frag[z], 0.0f);

          if (add)
            __syncthreads();

          for (int i = 0; i < 2; ++i)
            {
              wmma::load_matrix_sync(a_frag,
                                     &shape_data[i * 8],
                                     16 + skew_double);
              // #pragma unroll
              //               for (int t = 0; t < a_frag.num_elements; t++)
              //                 {
              //                   a_frag.x[t] =
              //                   wmma::__float_to_tf32(a_frag.x[t]);
              //                 }
              for (unsigned int z = 0; z < n_dofs_1d / 8; ++z)
                {
                  wmma::load_matrix_sync(
                    b_frag,
                    &in[(z * 8 + warpId) * 16 * (16 + skew_double) +
                        i * 8 * (16 + skew_double)],
                    16 + skew_double);
                  // #pragma unroll
                  //                   for (int t = 0; t < b_frag.num_elements;
                  //                   t++)
                  //                     {
                  //                       b_frag.x[t] =
                  //                       wmma::__float_to_tf32(b_frag.x[t]);
                  //                     }
                  wmma::mma_sync(c_frag[z], a_frag, b_frag, c_frag[z]);
                }
            }

          for (unsigned int z = 0; z < n_dofs_1d / 8; ++z)
            wmma::store_matrix_sync(
              &out[(z * 8 + warpId) * 16 * (16 + skew_double)],
              c_frag[z],
              16 + skew_double,
              wmma::mem_row_major);
        }
      else
        {
          constexpr int n_dofs_1d_padding = n_dofs_1d + skew_double;
          constexpr int stride            = n_dofs_1d * n_dofs_1d_padding;

          const int row = threadIdx.y;
          const int col = threadIdx.x % n_dofs_1d;

          float pval[n_dofs_1d];

          // kernel product: A kdot src, [N x N] * [N^dim, 1]
          for (unsigned int z = 0; z < n_dofs_1d; ++z)
            {
              pval[z] = 0;
              // #pragma unroll
              for (int k = 0; k < n_dofs_1d; ++k)
                {
                  const unsigned int shape_idx = z * n_dofs_1d_padding + k;
                  const unsigned int source_idx =
                    row * n_dofs_1d_padding + col + k * stride;

                  pval[z] += shape_data[shape_idx] * in[source_idx];
                }
            }

          for (unsigned int z = 0; z < n_dofs_1d; ++z)
            {
              const unsigned int destination_idx =
                row * n_dofs_1d_padding + col + z * stride;

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



  ////////////////////////////////////////////////////////////////////
  //////////////////// TPEvaluatorLaplace ////////////////////////////
  ////////////////////////////////////////////////////////////////////
  template <LaplaceVariant laplace_type,
            typename Number,
            int n_dofs_1d,
            int dim>
  struct TPEvaluatorLaplace
    : TPEvaluatorBase<TPEvaluatorLaplace<laplace_type, Number, n_dofs_1d, dim>,
                      n_dofs_1d,
                      Number,
                      laplace_type,
                      dim>
  {
    using TPEvaluatorBase<
      TPEvaluatorLaplace<laplace_type, Number, n_dofs_1d, dim>,
      n_dofs_1d,
      Number,
      laplace_type,
      dim>::apply;
    __device__ void
    vmult()
    {}
  };

  template <LaplaceVariant laplace_type, typename Number, int n_dofs_1d>
  struct TPEvaluatorLaplace<laplace_type, Number, n_dofs_1d, 2>
    : TPEvaluatorBase<TPEvaluatorLaplace<laplace_type, Number, n_dofs_1d, 2>,
                      n_dofs_1d,
                      Number,
                      laplace_type,
                      2>
  {
    using TPEvaluatorBase<
      TPEvaluatorLaplace<laplace_type, Number, n_dofs_1d, 2>,
      n_dofs_1d,
      Number,
      laplace_type,
      2>::apply;

    __device__ void
    vmult_impl(Number       *dst,
               const Number *src,
               const Number *mass_matrix,
               const Number *derivative_matrix,
               Number       *tmp)
    {
      constexpr unsigned int offset = n_dofs_1d * (n_dofs_1d + Util::padding);

      apply<0, false>(mass_matrix, src, tmp);
      __syncthreads();
      apply<1, false>(&derivative_matrix[offset], tmp, dst);
      __syncthreads();
      apply<0, false>(derivative_matrix, src, tmp);
      __syncthreads();
      apply<1, true>(&mass_matrix[offset], tmp, dst);
    }
  };

  template <LaplaceVariant laplace_type, typename Number, int n_dofs_1d>
  struct TPEvaluatorLaplace<laplace_type, Number, n_dofs_1d, 3>
    : TPEvaluatorBase<TPEvaluatorLaplace<laplace_type, Number, n_dofs_1d, 3>,
                      n_dofs_1d,
                      Number,
                      laplace_type,
                      3>
  {
    using TPEvaluatorBase<
      TPEvaluatorLaplace<laplace_type, Number, n_dofs_1d, 3>,
      n_dofs_1d,
      Number,
      laplace_type,
      3>::apply;

    __device__ void
    vmult_impl(Number       *dst,
               const Number *src,
               const Number *mass_matrix,
               const Number *derivative_matrix,
               Number       *tmp)
    {
      constexpr unsigned int local_dim =
        Util::pow(n_dofs_1d, 2) * (n_dofs_1d + Util::padding);
      constexpr unsigned int offset = n_dofs_1d * (n_dofs_1d + Util::padding);

      apply<0, false>(mass_matrix, src, &tmp[local_dim]);
      __syncthreads();
      apply<1, false>(&mass_matrix[offset], &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, false>(&derivative_matrix[offset * 2], tmp, dst);
      __syncthreads();
      apply<1, false>(&derivative_matrix[offset], &tmp[local_dim], tmp);
      __syncthreads();
      apply<0, false>(derivative_matrix, src, &tmp[local_dim]);
      __syncthreads();
      apply<1, true>(&mass_matrix[offset], &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, true>(&mass_matrix[offset * 2], tmp, dst);
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
               const Number *derivative_matrix,
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
               const Number *derivative_matrix,
               Number       *tmp)
    {
      apply<0, false>(mass_matrix, src, tmp);
      __syncthreads();
      apply<1, false, true>(derivative_matrix, tmp, dst);
      __syncthreads();
      apply<0, false>(derivative_matrix, src, tmp);
      __syncthreads();
      apply<1, false, true>(mass_matrix, tmp, dst);
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
               const Number *derivative_matrix,
               Number       *tmp)
    {
      constexpr unsigned int local_dim = Util::pow(n_dofs_1d, 3);

      apply<0, false>(mass_matrix, src, &tmp[local_dim]);
      __syncthreads();
      apply<1, false>(mass_matrix, &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, false, true>(derivative_matrix, tmp, dst);
      __syncthreads();
      apply<1, false>(derivative_matrix, &tmp[local_dim], tmp);
      __syncthreads();
      apply<0, false>(derivative_matrix, src, &tmp[local_dim]);
      __syncthreads();
      apply<1, true>(mass_matrix, &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, false, true>(mass_matrix, tmp, dst);
    }
  };


  template <typename Number, int n_dofs_1d>
  struct TPEvaluatorSmootherVmult<Number,
                                  n_dofs_1d,
                                  LaplaceVariant::TensorCore,
                                  2>
    : TPEvaluatorBase<TPEvaluatorSmootherVmult<Number,
                                               n_dofs_1d,
                                               LaplaceVariant::TensorCore,
                                               2>,
                      n_dofs_1d,
                      Number,
                      LaplaceVariant::TensorCore,
                      2>
  {
    using TPEvaluatorBase<TPEvaluatorSmootherVmult<Number,
                                                   n_dofs_1d,
                                                   LaplaceVariant::TensorCore,
                                                   2>,
                          n_dofs_1d,
                          Number,
                          LaplaceVariant::TensorCore,
                          2>::apply;

    __device__ void
    vmult_impl(Number       *dst,
               const Number *src,
               const Number *mass_matrix,
               const Number *derivative_matrix,
               Number       *tmp)
    {
      apply<0, false>(mass_matrix, src, tmp);
      __syncthreads();
      apply<1, false, false>(derivative_matrix, tmp, tmp);
      __syncthreads();
      dst[threadIdx.y * n_dofs_1d + threadIdx.x] -=
        tmp[threadIdx.y * n_dofs_1d + threadIdx.x];
      __syncthreads();
      apply<0, false>(derivative_matrix, src, tmp);
      __syncthreads();
      apply<1, false, false>(mass_matrix, tmp, tmp);
      __syncthreads();
      dst[threadIdx.y * n_dofs_1d + threadIdx.x] -=
        tmp[threadIdx.y * n_dofs_1d + threadIdx.x];
    }
  };



  template <typename Number,
            int             n_dofs_1d,
            SmootherVariant smoother_type,
            int             dim>
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
      apply<0, true>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, true>(eigenvectors, tmp, src);
      __syncthreads();
      src[threadIdx.y * n_dofs_1d + threadIdx.x % n_dofs_1d] /=
        (eigenvalues[threadIdx.y] + eigenvalues[threadIdx.x % n_dofs_1d]);
      __syncthreads();
      apply<0, false>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, false, false>(eigenvectors, tmp, dst);
    }

    template <int direction, bool contract_over_rows, bool add = false>
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
            contract_over_rows ? k * n_dofs_1d + row : row * n_dofs_1d + k;

          const unsigned int source_idx =
            (direction == 0) ? (col * n_dofs_1d + k) : (k * n_dofs_1d + col);

          pval += shape_data[shape_idx] * in[source_idx];
        }


      const unsigned int destination_idx =
        (direction == 0) ? (col * n_dofs_1d + row) : (row * n_dofs_1d + col);
      if (add)
        out[destination_idx] += pval;
      else
        out[destination_idx] = pval;
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

      apply<0, true>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, true>(eigenvectors, tmp, &tmp[local_dim]);
      __syncthreads();
      apply<2, true>(eigenvectors, &tmp[local_dim], tmp);
      __syncthreads();
      for (unsigned int z = 0; z < n_dofs_1d; ++z)
        {
          tmp[z * n_dofs_1d * n_dofs_1d + threadIdx.y * n_dofs_1d +
              threadIdx.x % n_dofs_1d] /=
            (eigenvalues[z] + eigenvalues[threadIdx.y] +
             eigenvalues[threadIdx.x % n_dofs_1d]);
        }
      __syncthreads();
      apply<0, false>(eigenvectors, tmp, &tmp[local_dim]);
      __syncthreads();
      apply<1, false>(eigenvectors, &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, false, false>(eigenvectors, tmp, dst);
    }

    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr unsigned int stride = n_dofs_1d * n_dofs_1d;

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



  template <typename Number, int n_dofs_1d>
  struct TPEvaluatorSmootherInv<Number, n_dofs_1d, SmootherVariant::FUSED_L, 2>
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

      unsigned int row = linear_tid / (n_dofs_1d - 2) + 1;
      unsigned int col = linear_tid % (n_dofs_1d - 2) + 1;

      apply<0, true>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, true>(eigenvectors, tmp, src);
      __syncthreads();
      if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
        src[row * n_dofs_1d + col] /= (eigenvalues[row] + eigenvalues[col]);
      __syncthreads();
      apply<0, false>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, false, false>(eigenvectors, tmp, dst);
    }

    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const unsigned int linear_tid =
        threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

      const unsigned int row = linear_tid / (n_dofs_1d - 2) + 1;
      const unsigned int col = linear_tid % (n_dofs_1d - 2) + 1;

      Number pval;

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
        {
          pval = 0;
          // #pragma unroll
          for (unsigned int k = 1; k < n_dofs_1d - 1; ++k)
            {
              const unsigned int shape_idx =
                contract_over_rows ? k * n_dofs_1d + row : row * n_dofs_1d + k;

              const unsigned int source_idx = (direction == 0) ?
                                                (col * n_dofs_1d + k) :
                                                (k * n_dofs_1d + col);

              pval += shape_data[shape_idx] * in[source_idx];
            }
        }

      if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
        {
          const unsigned int destination_idx = (direction == 0) ?
                                                 (col * n_dofs_1d + row) :
                                                 (row * n_dofs_1d + col);
          if (add)
            out[destination_idx] += pval;
          else
            out[destination_idx] = pval;
        }
    }
  };

  template <typename Number, int n_dofs_1d>
  struct TPEvaluatorSmootherInv<Number, n_dofs_1d, SmootherVariant::FUSED_L, 3>
  {
    __device__ void
    apply_inverse(Number       *dst,
                  Number       *src,
                  const Number *eigenvalues,
                  const Number *eigenvectors,
                  Number       *tmp)
    {
      constexpr unsigned int local_dim = Util::pow(n_dofs_1d, 3);
      const unsigned int     linear_tid =
        threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

      unsigned int row = linear_tid / (n_dofs_1d - 2) + 1;
      unsigned int col = linear_tid % (n_dofs_1d - 2) + 1;

      apply<0, true>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, true>(eigenvectors, tmp, &tmp[local_dim]);
      __syncthreads();
      apply<2, true>(eigenvectors, &tmp[local_dim], tmp);
      __syncthreads();
      if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
        for (unsigned int z = 1; z < n_dofs_1d - 1; ++z)
          {
            tmp[z * n_dofs_1d * n_dofs_1d + row * n_dofs_1d + col] /=
              (eigenvalues[z] + eigenvalues[row] + eigenvalues[col]);
          }
      __syncthreads();
      apply<0, false>(eigenvectors, tmp, &tmp[local_dim]);
      __syncthreads();
      apply<1, false>(eigenvectors, &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, false, false>(eigenvectors, tmp, dst);
    }

    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr unsigned int stride = n_dofs_1d * n_dofs_1d;
      const unsigned int     linear_tid =
        threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

      const unsigned int row = linear_tid / (n_dofs_1d - 2) + 1;
      const unsigned int col = linear_tid % (n_dofs_1d - 2) + 1;

      Number pval[n_dofs_1d - 2];

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
        for (unsigned int z = 1; z < n_dofs_1d - 1; ++z)
          {
            pval[z - 1] = 0;
            // #pragma unroll
            for (unsigned int k = 1; k < n_dofs_1d - 1; ++k)
              {
                const unsigned int shape_idx = contract_over_rows ?
                                                 k * n_dofs_1d + row :
                                                 row * n_dofs_1d + k;

                const unsigned int source_idx =
                  (direction == 0) ? (col * n_dofs_1d + k + z * stride) :
                  (direction == 1) ? (k * n_dofs_1d + col + z * stride) :
                                     (z * n_dofs_1d + col + k * stride);

                pval[z - 1] += shape_data[shape_idx] * in[source_idx];
              }
          }

      if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
        for (unsigned int z = 1; z < n_dofs_1d - 1; ++z)
          {
            const unsigned int destination_idx =
              (direction == 0) ? (col * n_dofs_1d + row + z * stride) :
              (direction == 1) ? (row * n_dofs_1d + col + z * stride) :
                                 (z * n_dofs_1d + col + row * stride);
            if (add)
              out[destination_idx] += pval[z - 1];
            else
              out[destination_idx] = pval[z - 1];
          }
    }
  };



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
      const unsigned int linear_tid =
        threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

      unsigned int row = linear_tid / (n_dofs_1d - 2);
      unsigned int col = linear_tid % (n_dofs_1d - 2);

      apply<0, true>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, true>(eigenvectors, tmp, src);
      __syncthreads();
      if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
        src[row * (n_dofs_1d - 2) + col] /=
          (eigenvalues[row] + eigenvalues[col]);
      __syncthreads();
      apply<0, false>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, false, false>(eigenvectors, tmp, dst);
    }

    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr unsigned int n_dofs_1d_i = n_dofs_1d - 2;
      
      constexpr int multiple = contract_over_rows ?
                                 n_dofs_1d_i :
                                 Util::calculate_multiple<n_dofs_1d_i, 16>();

      const unsigned int linear_tid =
        threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

      const unsigned int row = linear_tid / n_dofs_1d_i;
      const unsigned int col = linear_tid % n_dofs_1d_i;

      Number pval;

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      if (linear_tid < n_dofs_1d_i * n_dofs_1d_i)
        {
          pval = 0;
          // #pragma unroll
          for (unsigned int k = 0; k < n_dofs_1d_i; ++k)
            {
              const unsigned int shape_idx =
                contract_over_rows ?
                  ((direction == 0) ?
                     ((k + col / multiple) % n_dofs_1d_i) * n_dofs_1d_i + col :
                     k * n_dofs_1d_i + row) :
                  ((direction == 0) ?
                     col * n_dofs_1d_i + (k + col / multiple) % n_dofs_1d_i :
                     row * n_dofs_1d_i + k);

              const unsigned int source_idx =
                (direction == 0) ?
                  (row * n_dofs_1d_i + (k + col / multiple) % n_dofs_1d_i) :
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

      unsigned int row = linear_tid / (n_dofs_1d - 2);
      unsigned int col = linear_tid % (n_dofs_1d - 2);

      apply<0, true>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, true>(eigenvectors, tmp, src);
      __syncthreads();
      apply<2, true>(eigenvectors, src, tmp);
      __syncthreads();
      if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
        for (unsigned int z = 0; z < n_dofs_1d - 2; ++z)
          {
            tmp[z * n_dofs_1d * n_dofs_1d + row * (n_dofs_1d - 2) + col] /=
              (eigenvalues[z] + eigenvalues[row] + eigenvalues[col]);
          }
      __syncthreads();
      apply<0, false>(eigenvectors, tmp, src);
      __syncthreads();
      apply<1, false>(eigenvectors, src, tmp);
      __syncthreads();
      apply<2, false, false>(eigenvectors, tmp, dst);
    }

    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr unsigned int stride      = n_dofs_1d * n_dofs_1d;
      constexpr unsigned int n_dofs_1d_i = n_dofs_1d - 2;

      constexpr int multiple = contract_over_rows ?
                                 n_dofs_1d_i :
                                 Util::calculate_multiple<n_dofs_1d_i, 16>();

      const unsigned int linear_tid =
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
                    ((direction == 0) ?
                       ((k + col / multiple) % n_dofs_1d_i) * n_dofs_1d_i +
                         col :
                     (direction == 1) ? k * n_dofs_1d_i + row :
                                        k * n_dofs_1d_i + z) :
                    ((direction == 0) ?
                       col * n_dofs_1d_i + (k + col / multiple) % n_dofs_1d_i :
                     (direction == 1) ? row * n_dofs_1d_i + k :
                                        z * n_dofs_1d_i + k);

                const unsigned int source_idx =
                  (direction == 0) ?
                    (row * n_dofs_1d_i + (k + col / multiple) % n_dofs_1d_i +
                     z * stride) :
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
  evaluate_laplace(const unsigned int                local_patch,
                   SharedMemData<dim, Number, true> *shared_data)
  {
    constexpr unsigned int n_dofs_1d         = 2 * fe_degree + 2;
    constexpr unsigned int n_dofs_1d_padding = n_dofs_1d + Util::padding;
    constexpr unsigned int local_dim         = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int local_dim_padding =
      Util::pow(n_dofs_1d, dim - 1) * n_dofs_1d_padding;

    TPEvaluatorLaplace<laplace, Number, n_dofs_1d, dim> eval;
    __syncthreads();

    eval.vmult(&shared_data->local_dst[local_patch * local_dim_padding],
               &shared_data->local_src[local_patch * local_dim_padding],
               &shared_data->local_mass[local_patch * n_dofs_1d *
                                        n_dofs_1d_padding * dim],
               &shared_data->local_derivative[local_patch * n_dofs_1d *
                                              n_dofs_1d_padding * dim],
               &shared_data->tmp[local_patch * local_dim_padding * (dim - 1)]);
    __syncthreads();
  }

  template <int dim,
            int fe_degree,
            typename Number,
            LaplaceVariant  laplace,
            SmootherVariant smooth>
  __device__ void
  evaluate_smooth(
    const unsigned int                                             local_patch,
    SharedMemData<dim, Number, false>                             *shared_data,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data *gpu_data)
  {
    constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);

    TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace, dim> eval_vmult;
    TPEvaluatorSmootherInv<Number, n_dofs_1d, smooth, dim>    eval_inverse;
    __syncthreads();

    eval_vmult.vmult(&shared_data->local_src[local_patch * local_dim],
                     &shared_data->local_dst[local_patch * local_dim],
                     shared_data->local_mass,
                     shared_data->local_derivative,
                     &shared_data->tmp[local_patch * local_dim * (dim - 1)]);
    __syncthreads();

    const unsigned int linear_tid =
      threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

    if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
      {
        const unsigned int row = linear_tid / (n_dofs_1d - 2);
        const unsigned int col = linear_tid % (n_dofs_1d - 2);

        shared_data->local_mass[col + 1] = gpu_data->eigenvalues[col];
        shared_data->local_derivative[(row + 1) * n_dofs_1d + col + 1] =
          gpu_data->eigenvectors[row * (n_dofs_1d - 2) + col];
      }
    __syncthreads();

    eval_inverse.apply_inverse(
      &shared_data->local_dst[local_patch * local_dim],
      &shared_data->local_src[local_patch * local_dim],
      shared_data->local_mass,
      shared_data->local_derivative,
      &shared_data->tmp[local_patch * local_dim * (dim - 1)]);
    __syncthreads();
  }


  template <int dim,
            int fe_degree,
            typename Number,
            LaplaceVariant  laplace,
            SmootherVariant smooth>
  __device__ void
  evaluate_smooth_cf(
    const unsigned int                                             local_patch,
    SharedMemData<dim, Number, false>                             *shared_data,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data *gpu_data)
  {
    constexpr unsigned int n_dofs_1d   = 2 * fe_degree + 2;
    constexpr unsigned int n_dofs_1d_z = dim == 2 ? 1 : n_dofs_1d - 2;
    constexpr unsigned int local_dim   = Util::pow(n_dofs_1d, dim);

    TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace, dim> eval_vmult;
    TPEvaluatorSmootherInv<Number, n_dofs_1d, smooth, dim>    eval_inverse;
    __syncthreads();

    eval_vmult.vmult(&shared_data->local_src[local_patch * local_dim],
                     &shared_data->local_dst[local_patch * local_dim],
                     shared_data->local_mass,
                     shared_data->local_derivative,
                     &shared_data->tmp[local_patch * local_dim * (dim - 1)]);
    __syncthreads();

    const unsigned int linear_tid =
      threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

    if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
      {
        unsigned int row = linear_tid / (n_dofs_1d - 2);
        unsigned int col = linear_tid % (n_dofs_1d - 2);

        shared_data->local_mass[col] = gpu_data->eigenvalues[col];
        shared_data->local_derivative[row * (n_dofs_1d - 2) + col] =
          gpu_data->eigenvectors[row * (n_dofs_1d - 2) + col];
      }
    // __syncthreads();


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
    __syncthreads();

    eval_inverse.apply_inverse(
      &shared_data->tmp[local_patch * local_dim * 2],
      &shared_data->tmp[local_patch * local_dim * 2 + local_dim],
      shared_data->local_mass,
      shared_data->local_derivative,
      &shared_data->local_src[local_patch * local_dim]);
    __syncthreads();
  }


  template <int dim, int fe_degree, typename Number, SmootherVariant smooth>
  __device__ void
  evaluate_smooth_inv(const unsigned int                 local_patch,
                      SharedMemData<dim, Number, false> *shared_data)
  {
    constexpr unsigned int n_dofs_1d = 2 * fe_degree;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);

    TPEvaluatorSmootherInv<Number, n_dofs_1d, smooth, dim> eval;
    __syncthreads();

    eval.apply_inverse(&shared_data->local_dst[local_patch * local_dim],
                       &shared_data->local_src[local_patch * local_dim],
                       shared_data->local_mass,
                       shared_data->local_derivative,
                       &shared_data->tmp[local_patch * local_dim * (dim - 1)]);
    __syncthreads();
  }

} // namespace PSMF


#endif // CUDA_EVALUATE_CUH