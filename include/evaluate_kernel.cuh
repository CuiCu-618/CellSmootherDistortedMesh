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

#include "patch_base.cuh"

namespace PSMF
{

  /**
   * Compute residual based on tensor product structure.
   */
  template <int kernel_size,
            typename Number,
            int             dim,
            SmootherVariant kernel,
            DoFLayout       dof_layout>
  struct TPEvaluator_vmult;

  /**
   * local solver based on tensor product structure.
   */
  template <int kernel_size,
            typename Number,
            int             dim,
            SmootherVariant kernel,
            DoFLayout       dof_layout>
  struct TPEvaluator_inverse;


  template <int kernel_size, typename Number>
  struct TPEvaluator_vmult<kernel_size,
                           Number,
                           2,
                           SmootherVariant::SEPERATE,
                           DoFLayout::Q>
  {
    /**
     *  Constructor.
     */
    __device__
    TPEvaluator_vmult()
    {}

    /**
     * Vector multication.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *temp)
    {
      apply<0, false>(mass_matrix, src, temp);
      __syncthreads();
      apply<1, false, true>(derivative_matrix, temp, dst);
      __syncthreads();
      apply<0, false>(derivative_matrix, src, temp);
      __syncthreads();
      apply<1, false, true>(mass_matrix, temp, dst);
    }

    /**
     * apply 1d @p shape_data vector to @p in.
     */
    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int row = threadIdx.y;
      const int col = threadIdx.x % kernel_size;

      Number pval = 0;

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (int k = 0; k < kernel_size; ++k)
        {
          const unsigned int shape_idx  = row * kernel_size + k;
          const unsigned int source_idx = (direction == 0) ?
                                            (col * kernel_size + k) :
                                            (k * kernel_size + col);

          pval += shape_data[shape_idx] * in[source_idx];
        }

      const unsigned int destination_idx = (direction == 0) ?
                                             (col * kernel_size + row) :
                                             (row * kernel_size + col);

      if (add)
        out[destination_idx] += pval;
      if (sub)
        out[destination_idx] -= pval;
      else
        out[destination_idx] = pval;
    }
  };

  template <int kernel_size, typename Number>
  struct TPEvaluator_vmult<kernel_size,
                           Number,
                           3,
                           SmootherVariant::SEPERATE,
                           DoFLayout::Q>
  {
    /**
     *  Constructor.
     */
    __device__
    TPEvaluator_vmult()
    {}

    /**
     * Vector multication.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *temp)
    {
      constexpr unsigned int local_dim = Util::pow(kernel_size, 3);

      apply<0, false>(mass_matrix, src, &temp[local_dim]);
      __syncthreads();
      apply<1, false>(mass_matrix, &temp[local_dim], temp);
      __syncthreads();
      apply<2, false, true>(derivative_matrix, temp, dst);
      __syncthreads();
      apply<1, false>(derivative_matrix, &temp[local_dim], temp);
      __syncthreads();
      apply<0, false>(derivative_matrix, src, &temp[local_dim]);
      __syncthreads();
      apply<1, true>(mass_matrix, &temp[local_dim], temp);
      __syncthreads();
      apply<2, false, true>(mass_matrix, temp, dst);
    }

    /**
     * apply 1d @p shape_data vector to @p in.
     */
    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int stride = kernel_size * kernel_size;

      const unsigned int row = threadIdx.y;
      const unsigned int col = threadIdx.x % kernel_size;

      Number pval[kernel_size];

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      for (unsigned int z = 0; z < kernel_size; ++z)
        {
          pval[z] = 0;
          // #pragma unroll
          for (unsigned int k = 0; k < kernel_size; ++k)
            {
              const unsigned int shape_idx = row * kernel_size + k;

              const unsigned int source_idx =
                (direction == 0) ? (col * kernel_size + k + z * stride) :
                (direction == 1) ? (k * kernel_size + col + z * stride) :
                                   (z * kernel_size + col + k * stride);

              pval[z] += shape_data[shape_idx] * in[source_idx];
            }
        }

      for (unsigned int z = 0; z < kernel_size; ++z)
        {
          const unsigned int destination_idx =
            (direction == 0) ? (col * kernel_size + row + z * stride) :
            (direction == 1) ? (row * kernel_size + col + z * stride) :
                               (z * kernel_size + col + row * stride);

          if (add)
            out[destination_idx] += pval[z];
          else if (sub)
            out[destination_idx] -= pval[z];
          else
            out[destination_idx] = pval[z];
        }
    }
  };

  template <int kernel_size, typename Number>
  struct TPEvaluator_vmult<kernel_size,
                           Number,
                           2,
                           SmootherVariant::FUSED,
                           DoFLayout::Q>
  {
    /**
     *  Constructor.
     */
    __device__
    TPEvaluator_vmult()
    {}

    /**
     * Vector multication.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *temp)
    {
      apply<0, false>(mass_matrix, src, temp);
      __syncthreads();
      apply<1, false, true>(derivative_matrix, temp, dst);
      __syncthreads();
      apply<0, false>(derivative_matrix, src, temp);
      __syncthreads();
      apply<1, false, true>(mass_matrix, temp, dst);
    }

    /**
     * apply 1d @p shape_data vector to @p in.
     */
    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int row = threadIdx.y;
      const int col = threadIdx.x % kernel_size;

      Number pval = 0;

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (int k = 0; k < kernel_size; ++k)
        {
          const unsigned int shape_idx  = row * kernel_size + k;
          const unsigned int source_idx = (direction == 0) ?
                                            (col * kernel_size + k) :
                                            (k * kernel_size + col);

          pval += shape_data[shape_idx] * in[source_idx];
        }

      const unsigned int destination_idx = (direction == 0) ?
                                             (col * kernel_size + row) :
                                             (row * kernel_size + col);

      if (add)
        out[destination_idx] += pval;
      if (sub)
        out[destination_idx] -= pval;
      else
        out[destination_idx] = pval;
    }
  };

  template <int kernel_size, typename Number>
  struct TPEvaluator_vmult<kernel_size,
                           Number,
                           3,
                           SmootherVariant::FUSED,
                           DoFLayout::Q>
  {
    /**
     *  Constructor.
     */
    __device__
    TPEvaluator_vmult()
    {}

    /**
     * Vector multication.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *temp)
    {
      constexpr unsigned int local_dim = Util::pow(kernel_size, 3);

      apply<0, false>(mass_matrix, src, &temp[local_dim]);
      __syncthreads();
      apply<1, false>(mass_matrix, &temp[local_dim], temp);
      __syncthreads();
      apply<2, false, true>(derivative_matrix, temp, dst);
      __syncthreads();
      apply<1, false>(derivative_matrix, &temp[local_dim], temp);
      __syncthreads();
      apply<0, false>(derivative_matrix, src, &temp[local_dim]);
      __syncthreads();
      apply<1, true>(mass_matrix, &temp[local_dim], temp);
      __syncthreads();
      apply<2, false, true>(mass_matrix, temp, dst);
    }

    /**
     * apply 1d @p shape_data vector to @p in.
     */
    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int stride = kernel_size * kernel_size;

      const unsigned int row = threadIdx.y;
      const unsigned int col = threadIdx.x % kernel_size;

      Number pval[kernel_size];

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      for (unsigned int z = 0; z < kernel_size; ++z)
        {
          pval[z] = 0;
          // #pragma unroll
          for (unsigned int k = 0; k < kernel_size; ++k)
            {
              const unsigned int shape_idx = row * kernel_size + k;

              const unsigned int source_idx =
                (direction == 0) ? (col * kernel_size + k + z * stride) :
                (direction == 1) ? (k * kernel_size + col + z * stride) :
                                   (z * kernel_size + col + k * stride);

              pval[z] += shape_data[shape_idx] * in[source_idx];
            }
        }

      for (unsigned int z = 0; z < kernel_size; ++z)
        {
          const unsigned int destination_idx =
            (direction == 0) ? (col * kernel_size + row + z * stride) :
            (direction == 1) ? (row * kernel_size + col + z * stride) :
                               (z * kernel_size + col + row * stride);

          if (add)
            out[destination_idx] += pval[z];
          else if (sub)
            out[destination_idx] -= pval[z];
          else
            out[destination_idx] = pval[z];
        }
    }
  };


  template <int kernel_size, typename Number>
  struct TPEvaluator_inverse<kernel_size,
                             Number,
                             2,
                             SmootherVariant::SEPERATE,
                             DoFLayout::Q>
  {
    /**
     *  Constructor.
     */
    __device__
    TPEvaluator_inverse()
    {}

    /**
     * Apply inverse to @p src.
     */
    __device__ void
    apply_inverse(Number       *dst,
                  Number       *src,
                  const Number *eigenvalues,
                  const Number *eigenvectors,
                  Number       *temp)
    {
      const unsigned int local_tid_x = threadIdx.x % kernel_size;

      apply<0, true>(eigenvectors, src, temp);
      __syncthreads();
      apply<1, true>(eigenvectors, temp, src);
      __syncthreads();
      src[threadIdx.y * kernel_size + local_tid_x] /=
        (eigenvalues[threadIdx.y] + eigenvalues[local_tid_x]);
      __syncthreads();
      apply<0, false>(eigenvectors, src, temp);
      __syncthreads();
      apply<1, false, true>(eigenvectors, temp, dst);
    }

    /**
     * apply 1d @p shape_data vector to @p in.
     */
    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int row = threadIdx.y;
      const int col = threadIdx.x % kernel_size;

      Number pval = 0;

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (unsigned int k = 0; k < kernel_size; ++k)
        {
          const unsigned int shape_idx =
            contract_over_rows ? k * kernel_size + row : row * kernel_size + k;

          const unsigned int source_idx = (direction == 0) ?
                                            (col * kernel_size + k) :
                                            (k * kernel_size + col);

          pval += shape_data[shape_idx] * in[source_idx];
        }


      const unsigned int destination_idx = (direction == 0) ?
                                             (col * kernel_size + row) :
                                             (row * kernel_size + col);
      if (add)
        out[destination_idx] += pval;
      else
        out[destination_idx] = pval;
    }
  };

  template <int kernel_size, typename Number>
  struct TPEvaluator_inverse<kernel_size,
                             Number,
                             3,
                             SmootherVariant::SEPERATE,
                             DoFLayout::Q>
  {
    /**
     *  Constructor.
     */
    __device__
    TPEvaluator_inverse()
    {}

    /**
     * Apply inverse to @p src.
     */
    __device__ void
    apply_inverse(Number       *dst,
                  Number       *src,
                  const Number *eigenvalues,
                  const Number *eigenvectors,
                  Number       *temp)
    {
      constexpr unsigned int local_dim   = Util::pow(kernel_size, 3);
      const unsigned int     local_tid_x = threadIdx.x % kernel_size;

      apply<0, true>(eigenvectors, src, temp);
      __syncthreads();
      apply<1, true>(eigenvectors, temp, src);
      __syncthreads();
      apply<2, true>(eigenvectors, src, temp);
      __syncthreads();
      for (unsigned int z = 0; z < kernel_size; ++z)
        {
          temp[z * kernel_size * kernel_size + threadIdx.y * kernel_size +
               local_tid_x] /= (eigenvalues[z] + eigenvalues[threadIdx.y] +
                                eigenvalues[local_tid_x]);
        }
      __syncthreads();
      apply<0, false>(eigenvectors, temp, src);
      __syncthreads();
      apply<1, false>(eigenvectors, src, temp);
      __syncthreads();
      apply<2, false, true>(eigenvectors, temp, dst);
    }

    /**
     * apply 1d @p shape_data vector to @p in.
     */
    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int stride = kernel_size * kernel_size;

      const unsigned int row = threadIdx.y;
      const unsigned int col = threadIdx.x % kernel_size;

      Number pval[kernel_size];

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      for (unsigned int z = 0; z < kernel_size; ++z)
        {
          pval[z] = 0;
          // #pragma unroll
          for (unsigned int k = 0; k < kernel_size; ++k)
            {
              const unsigned int shape_idx = contract_over_rows ?
                                               k * kernel_size + row :
                                               row * kernel_size + k;

              const unsigned int source_idx =
                (direction == 0) ? (col * kernel_size + k + z * stride) :
                (direction == 1) ? (k * kernel_size + col + z * stride) :
                                   (z * kernel_size + col + k * stride);

              pval[z] += shape_data[shape_idx] * in[source_idx];
            }
        }

      for (unsigned int z = 0; z < kernel_size; ++z)
        {
          const unsigned int destination_idx =
            (direction == 0) ? (col * kernel_size + row + z * stride) :
            (direction == 1) ? (row * kernel_size + col + z * stride) :
                               (z * kernel_size + col + row * stride);
          if (add)
            out[destination_idx] += pval[z];
          else
            out[destination_idx] = pval[z];
        }
    }
  };



  template <int kernel_size, typename Number>
  struct TPEvaluator_inverse<kernel_size,
                             Number,
                             2,
                             SmootherVariant::FUSED,
                             DoFLayout::Q>
  {
    /**
     *  Constructor.
     */
    __device__
    TPEvaluator_inverse()
    {}

    /**
     * Apply inverse to @p src.
     */
    __device__ void
    apply_inverse(Number       *dst,
                  Number       *src,
                  const Number *eigenvalues,
                  const Number *eigenvectors,
                  Number       *temp)
    {
      const unsigned int linear_tid =
        threadIdx.x % kernel_size + threadIdx.y * kernel_size;

      unsigned int row = linear_tid / (kernel_size - 2) + 1;
      unsigned int col = linear_tid % (kernel_size - 2) + 1;

      apply<0, true>(eigenvectors, src, temp);
      __syncthreads();
      apply<1, true>(eigenvectors, temp, src);
      __syncthreads();
      if (linear_tid < (kernel_size - 2) * (kernel_size - 2))
        src[row * kernel_size + col] /= (eigenvalues[row] + eigenvalues[col]);
      __syncthreads();
      apply<0, false>(eigenvectors, src, temp);
      __syncthreads();
      apply<1, false, true>(eigenvectors, temp, dst);
    }

    /**
     * apply 1d @p shape_data vector to @p in.
     */
    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const unsigned int linear_tid =
        threadIdx.x % kernel_size + threadIdx.y * kernel_size;

      const unsigned int row = linear_tid / (kernel_size - 2) + 1;
      const unsigned int col = linear_tid % (kernel_size - 2) + 1;

      Number pval;

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      if (linear_tid < (kernel_size - 2) * (kernel_size - 2))
        {
          pval = 0;
          // #pragma unroll
          for (unsigned int k = 1; k < kernel_size - 1; ++k)
            {
              const unsigned int shape_idx = contract_over_rows ?
                                               k * kernel_size + row :
                                               row * kernel_size + k;

              const unsigned int source_idx = (direction == 0) ?
                                                (col * kernel_size + k) :
                                                (k * kernel_size + col);

              pval += shape_data[shape_idx] * in[source_idx];
            }
        }

      if (linear_tid < (kernel_size - 2) * (kernel_size - 2))
        {
          const unsigned int destination_idx = (direction == 0) ?
                                                 (col * kernel_size + row) :
                                                 (row * kernel_size + col);
          if (add)
            out[destination_idx] += pval;
          else
            out[destination_idx] = pval;
        }
    }
  };

  template <int kernel_size, typename Number>
  struct TPEvaluator_inverse<kernel_size,
                             Number,
                             3,
                             SmootherVariant::FUSED,
                             DoFLayout::Q>
  {
    /**
     *  Constructor.
     */
    __device__
    TPEvaluator_inverse()
    {}

    /**
     * Apply inverse to @p src.
     */
    __device__ void
    apply_inverse(Number       *dst,
                  Number       *src,
                  const Number *eigenvalues,
                  const Number *eigenvectors,
                  Number       *temp)
    {
      constexpr unsigned int local_dim = Util::pow(kernel_size, 3);
      const unsigned int     linear_tid =
        threadIdx.x % kernel_size + threadIdx.y * kernel_size;

      unsigned int row = linear_tid / (kernel_size - 2) + 1;
      unsigned int col = linear_tid % (kernel_size - 2) + 1;

      apply<0, true>(eigenvectors, src, temp);
      __syncthreads();
      apply<1, true>(eigenvectors, temp, &temp[local_dim]);
      __syncthreads();
      apply<2, true>(eigenvectors, &temp[local_dim], temp);
      __syncthreads();
      if (linear_tid < (kernel_size - 2) * (kernel_size - 2))
        for (unsigned int z = 1; z < kernel_size - 1; ++z)
          {
            temp[z * kernel_size * kernel_size + row * kernel_size + col] /=
              (eigenvalues[z] + eigenvalues[row] + eigenvalues[col]);
          }
      __syncthreads();
      apply<0, false>(eigenvectors, temp, &temp[local_dim]);
      __syncthreads();
      apply<1, false>(eigenvectors, &temp[local_dim], temp);
      __syncthreads();
      apply<2, false, true>(eigenvectors, temp, dst);
    }

    /**
     * apply 1d @p shape_data vector to @p in.
     */
    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int      stride = kernel_size * kernel_size;
      const unsigned int linear_tid =
        threadIdx.x % kernel_size + threadIdx.y * kernel_size;

      const unsigned int row = linear_tid / (kernel_size - 2) + 1;
      const unsigned int col = linear_tid % (kernel_size - 2) + 1;

      Number pval[kernel_size];

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      if (linear_tid < (kernel_size - 2) * (kernel_size - 2))
        for (unsigned int z = 1; z < kernel_size - 1; ++z)
          {
            pval[z] = 0;
            // #pragma unroll
            for (unsigned int k = 1; k < kernel_size - 1; ++k)
              {
                const unsigned int shape_idx = contract_over_rows ?
                                                 k * kernel_size + row :
                                                 row * kernel_size + k;

                const unsigned int source_idx =
                  (direction == 0) ? (col * kernel_size + k + z * stride) :
                  (direction == 1) ? (k * kernel_size + col + z * stride) :
                                     (z * kernel_size + col + k * stride);

                pval[z] += shape_data[shape_idx] * in[source_idx];
              }
          }

      if (linear_tid < (kernel_size - 2) * (kernel_size - 2))
        for (unsigned int z = 1; z < kernel_size - 1; ++z)
          {
            const unsigned int destination_idx =
              (direction == 0) ? (col * kernel_size + row + z * stride) :
              (direction == 1) ? (row * kernel_size + col + z * stride) :
                                 (z * kernel_size + col + row * stride);
            if (add)
              out[destination_idx] += pval[z];
            else
              out[destination_idx] = pval[z];
          }
    }
  };



  /**
   * @brief Functor. Local smoother based on tensor product structure.
   *
   * @tparam dim
   * @tparam fe_degree
   * @tparam Number
   * @tparam kernel
   * @tparam dof_layout
   */
  template <int dim,
            int fe_degree,
            typename Number,
            SmootherVariant kernel,
            DoFLayout       dof_layout>
  class LocalSmoother;

  template <int dim, int fe_degree, typename Number>
  class LocalSmoother<dim,
                      fe_degree,
                      Number,
                      SmootherVariant::SEPERATE,
                      DoFLayout::Q>
  {
  public:
    static constexpr unsigned int n_dofs_1d = 2 * fe_degree + 1;

    LocalSmoother()
      : ndofs_per_dim(0)
    {}

    LocalSmoother(unsigned int ndofs_per_dim)
      : ndofs_per_dim(ndofs_per_dim)
    {}

    __device__ inline unsigned int
    get_ndofs()
    {
      return ndofs_per_dim;
    }

    __device__ void
    operator()(
      const unsigned int                                     patch,
      const typename LevelVertexPatch<dim,
                                      fe_degree,
                                      Number,
                                      SmootherVariant::SEPERATE,
                                      DoFLayout::Q>::Data   *gpu_data,
      SharedMemData<dim, Number, SmootherVariant::SEPERATE> *shared_data) const
    {
      constexpr unsigned int local_dim   = Util::pow(n_dofs_1d, dim);

      TPEvaluator_vmult<n_dofs_1d,
                        Number,
                        dim,
                        SmootherVariant::SEPERATE,
                        DoFLayout::Q>
        eval;
      __syncthreads();

      eval.vmult(&shared_data->local_src[patch * local_dim],
                 &shared_data->local_dst[patch * local_dim],
                 shared_data->local_mass,
                 shared_data->local_derivative,
                 &shared_data->temp[patch * local_dim * (dim - 1)]);
      __syncthreads();
    }

    unsigned int ndofs_per_dim;
  };


  template <int dim,
            int fe_degree,
            typename Number,
            SmootherVariant kernel,
            DoFLayout       dof_layout>
  class LocalSmoother_inverse
  {
  public:
    static constexpr unsigned int n_dofs_1d = 2 * fe_degree - 1;

    LocalSmoother_inverse()
      : ndofs_per_dim(0)
    {}

    LocalSmoother_inverse(unsigned int ndofs_per_dim)
      : ndofs_per_dim(ndofs_per_dim)
    {}

    __device__ inline unsigned int
    get_ndofs()
    {
      return ndofs_per_dim;
    }

    __device__ void
    operator()(const unsigned int                                 patch,
               const typename LevelVertexPatch<dim,
                                               fe_degree,
                                               Number,
                                               kernel,
                                               dof_layout>::Data *gpu_data,
               SharedMemData<dim, Number, kernel> *shared_data) const
    {
      const unsigned int local_dim   = Util::pow(n_dofs_1d, dim);

      TPEvaluator_inverse<n_dofs_1d, Number, dim, kernel, dof_layout> eval;
      __syncthreads();

      // local inverse
      eval.apply_inverse(&shared_data->local_dst[patch * local_dim],
                         &shared_data->local_src[patch * local_dim],
                         shared_data->local_mass,
                         shared_data->local_derivative,
                         &shared_data->temp[patch * local_dim]);
      __syncthreads();
    }

    unsigned int ndofs_per_dim;
  };



  template <int dim, int fe_degree, typename Number>
  class LocalSmoother<dim,
                      fe_degree,
                      Number,
                      SmootherVariant::FUSED,
                      DoFLayout::Q>
  {
  public:
    static constexpr unsigned int n_dofs_1d = 2 * fe_degree + 1;

    LocalSmoother()
      : ndofs_per_dim(0)
    {}

    LocalSmoother(unsigned int ndofs_per_dim)
      : ndofs_per_dim(ndofs_per_dim)
    {}

    __device__ inline unsigned int
    get_ndofs()
    {
      return ndofs_per_dim;
    }

    __device__ void
    operator()(
      const unsigned int                                   patch,
      const typename LevelVertexPatch<dim,
                                      fe_degree,
                                      Number,
                                      SmootherVariant::FUSED,
                                      DoFLayout::Q>::Data *gpu_data,
      SharedMemData<dim, Number, SmootherVariant::FUSED>  *shared_data) const
    {
      constexpr unsigned int local_dim   = Util::pow(n_dofs_1d, dim);
      const unsigned int     local_tid_x = threadIdx.x % n_dofs_1d;

      TPEvaluator_vmult<n_dofs_1d,
                        Number,
                        dim,
                        SmootherVariant::FUSED,
                        DoFLayout::Q>
        eval_vmult;
      TPEvaluator_inverse<n_dofs_1d,
                          Number,
                          dim,
                          SmootherVariant::FUSED,
                          DoFLayout::Q>
        eval_inverse;
      __syncthreads();

      eval_vmult.vmult(&shared_data->local_src[patch * local_dim],
                       &shared_data->local_dst[patch * local_dim],
                       shared_data->local_mass,
                       shared_data->local_derivative,
                       &shared_data->temp[patch * local_dim * (dim - 1)]);
      __syncthreads();

      unsigned int linear_tid = local_tid_x + threadIdx.y * n_dofs_1d;

      if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
        {
          int row = linear_tid / (n_dofs_1d - 2);
          int col = linear_tid % (n_dofs_1d - 2);

          shared_data->local_mass[col + 1] = gpu_data->eigenvalues[col];
          shared_data->local_derivative[(row + 1) * n_dofs_1d + col + 1] =
            gpu_data->eigenvectors[row * (n_dofs_1d - 2) + col];
        }
      __syncthreads();

      eval_inverse.apply_inverse(
        &shared_data->local_dst[patch * local_dim],
        &shared_data->local_src[patch * local_dim],
        shared_data->local_mass,
        shared_data->local_derivative,
        &shared_data->temp[patch * local_dim * (dim - 1)]);
      __syncthreads();
    }

    unsigned int ndofs_per_dim;
  };

} // namespace PSMF


#endif // CUDA_EVALUATE_CUH