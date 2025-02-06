/**
 * @file loop_kernel.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief collection of global functions
 * @version 1.0
 * @date 2022-12-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef LOOP_KERNEL_CUH
#define LOOP_KERNEL_CUH

#include <type_traits>

#include "evaluate_kernel.cuh"
#include "patch_base.cuh"

namespace PSMF
{

  /**
   * Shared data for laplace operator kernel.
   * We have to use the following thing to avoid
   * a compile time error since @tparam Number
   * could be double and float at same time.
   */
  extern __shared__ double data_d[];
  extern __shared__ float  data_f[];

  template <typename Number>
  __device__ inline Number *
  get_shared_data_ptr();

  template <>
  __device__ inline double *
  get_shared_data_ptr()
  {
    return data_d;
  }

  template <>
  __device__ inline float *
  get_shared_data_ptr()
  {
    return data_f;
  }

#ifdef TENSORCORE
  template <int dim, int fe_degree, typename Number, LaplaceVariant laplace>
  __global__ void
  laplace_kernel_basic(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int n_dofs_z  = dim == 2 ? 1 : 2 * fe_degree + 1;

    const unsigned int n_dofs_per_dim  = gpu_data.n_dofs_per_dim;
    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedDataOp<dim, Number> shared_data(get_shared_data_ptr<Number>(),
                                          patch_per_block,
                                          n_dofs_1d,
                                          local_dim);

    if (patch < gpu_data.n_patches)
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            shared_data
              .local_mass[(local_patch * dim + d) * n_dofs_1d * n_dofs_1d +
                          threadIdx.y * n_dofs_1d + local_tid_x] =
              gpu_data.laplace_mass_1d[gpu_data.patch_type[patch * dim + d] *
                                         n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x];

            shared_data
              .local_laplace[(local_patch * dim + d) * n_dofs_1d * n_dofs_1d +
                             threadIdx.y * n_dofs_1d + local_tid_x] =
              gpu_data.laplace_stiff_1d[gpu_data.patch_type[patch * dim + d] *
                                          n_dofs_1d * n_dofs_1d +
                                        threadIdx.y * n_dofs_1d + local_tid_x];

            shared_data
              .local_bilaplace[(local_patch * dim + d) * n_dofs_1d * n_dofs_1d +
                               threadIdx.y * n_dofs_1d + local_tid_x] =
              gpu_data
                .bilaplace_stiff_1d[gpu_data.patch_type[patch * dim + d] *
                                      n_dofs_1d * n_dofs_1d +
                                    threadIdx.y * n_dofs_1d + local_tid_x];
          }

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x;

            const unsigned int global_dof_indices =
              z * n_dofs_per_dim * n_dofs_per_dim +
              threadIdx.y * n_dofs_per_dim + local_tid_x +
              gpu_data.first_dof[patch];

            if (threadIdx.y < 2 * fe_degree + 1 &&
                local_tid_x < 2 * fe_degree + 1)
              shared_data.local_src[index] = src[global_dof_indices];
            else
              shared_data.local_src[index] = 0;

            shared_data.local_dst[index] = 0.;
          }

        evaluate_laplace<dim, fe_degree, Number, laplace>(local_patch,
                                                          &shared_data);

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x;

            const unsigned int global_dof_indices =
              z * n_dofs_per_dim * n_dofs_per_dim +
              threadIdx.y * n_dofs_per_dim + local_tid_x +
              gpu_data.first_dof[patch];

            if (threadIdx.y < 2 * fe_degree + 1 &&
                local_tid_x < 2 * fe_degree + 1)
              atomicAdd(&dst[global_dof_indices], shared_data.local_dst[index]);
          }
      }
  }


#else
  template <int dim, int fe_degree, typename Number, LaplaceVariant laplace>
  __global__ void
  laplace_kernel_basic(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d = 2 * fe_degree + 1;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int n_dofs_per_dim  = gpu_data.n_dofs_per_dim;
    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedDataOp<dim, Number> shared_data(get_shared_data_ptr<Number>(),
                                          patch_per_block,
                                          n_dofs_1d,
                                          local_dim);

    if (patch < gpu_data.n_patches)
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            shared_data
              .local_mass[(local_patch * dim + d) * n_dofs_1d * n_dofs_1d +
                          threadIdx.y * n_dofs_1d + local_tid_x] =
              gpu_data.laplace_mass_1d[gpu_data.patch_type[patch * dim + d] *
                                         n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x];

            shared_data
              .local_laplace[(local_patch * dim + d) * n_dofs_1d * n_dofs_1d +
                             threadIdx.y * n_dofs_1d + local_tid_x] =
              gpu_data.laplace_stiff_1d[gpu_data.patch_type[patch * dim + d] *
                                          n_dofs_1d * n_dofs_1d +
                                        threadIdx.y * n_dofs_1d + local_tid_x];

            shared_data
              .local_bilaplace[(local_patch * dim + d) * n_dofs_1d * n_dofs_1d +
                               threadIdx.y * n_dofs_1d + local_tid_x] =
              gpu_data
                .bilaplace_stiff_1d[gpu_data.patch_type[patch * dim + d] *
                                      n_dofs_1d * n_dofs_1d +
                                    threadIdx.y * n_dofs_1d + local_tid_x];
          }

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x;

            const unsigned int global_dof_indices =
              z * n_dofs_per_dim * n_dofs_per_dim +
              threadIdx.y * n_dofs_per_dim + local_tid_x +
              gpu_data.first_dof[patch];

            shared_data.local_src[index] = src[global_dof_indices];

            shared_data.local_dst[index] = 0.;
          }

        evaluate_laplace<dim, fe_degree, Number, laplace>(local_patch,
                                                          &shared_data);

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x;

            const unsigned int global_dof_indices =
              z * n_dofs_per_dim * n_dofs_per_dim +
              threadIdx.y * n_dofs_per_dim + local_tid_x +
              gpu_data.first_dof[patch];

            atomicAdd(&dst[global_dof_indices], shared_data.local_dst[index]);
          }
      }
  }
#endif

  template <int dim,
            int fe_degree,
            typename Number,
            LocalSolverVariant local_solver>
  __global__
    typename std::enable_if<local_solver == LocalSolverVariant::Exact>::type
    loop_kernel_global(
      const Number                                                 *src,
      Number                                                       *dst,
      const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d = 2 * fe_degree - 1;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int n_dofs_per_dim  = gpu_data.n_dofs_per_dim;
    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;
    const unsigned int linear_tid_x = threadIdx.y * n_dofs_1d + local_tid_x;

    SharedDataSmoother<dim, Number, SmootherVariant::GLOBAL, local_solver>
      shared_data(get_shared_data_ptr<Number>(),
                  patch_per_block,
                  n_dofs_1d,
                  local_dim);

    if (patch < gpu_data.n_patches)
      {
        // #pragma unroll
        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x;

            const unsigned int global_dof_indices =
              (z + dim - 2) * n_dofs_per_dim * n_dofs_per_dim +
              (threadIdx.y + 1) * n_dofs_per_dim + local_tid_x + 1 +
              gpu_data.first_dof[patch];

            shared_data.local_src[index] = src[global_dof_indices];

#if SCHWARZTYPE == 2
            shared_data.local_dst[index] = 0;
#else
            shared_data.local_dst[index] = dst[global_dof_indices];
#endif
          }

        constexpr unsigned int matrix_size = Util::pow(local_dim, 2);
        unsigned int           patch_type  = 0;
        for (unsigned int d = 0; d < dim; ++d)
          patch_type += gpu_data.patch_type[patch * dim + d] * Util::pow(3, d);
        for (unsigned int row = 0; row < local_dim; ++row)
          {
            for (unsigned int z = 0; z < n_dofs_z; ++z)
              {
                auto val =
                  gpu_data
                    .eigenvalues[patch_type * matrix_size + row * local_dim +
                                 z * n_dofs_1d * n_dofs_1d + linear_tid_x] *
                  shared_data
                    .local_src[local_patch * local_dim +
                               z * n_dofs_1d * n_dofs_1d + linear_tid_x];

                atomicAdd(&shared_data.local_dst[local_patch * local_dim + row],
                          val);
              }
          }

        __syncthreads();

        // #pragma unroll
        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x;

            const unsigned int global_dof_indices =
              (z + dim - 2) * n_dofs_per_dim * n_dofs_per_dim +
              (threadIdx.y + 1) * n_dofs_per_dim + local_tid_x + 1 +
              gpu_data.first_dof[patch];

#if SCHWARZTYPE == 2
            atomicAdd(&dst[global_dof_indices],
                      shared_data.local_dst[index] * gpu_data.relaxation);
#else
            dst[global_dof_indices] =
              shared_data.local_dst[index] * gpu_data.relaxation;
#endif
          }
      }
  }

  template <int dim,
            int fe_degree,
            typename Number,
            LocalSolverVariant local_solver>
  __global__
    typename std::enable_if<local_solver != LocalSolverVariant::Exact>::type
    loop_kernel_global(
      const Number                                                 *src,
      Number                                                       *dst,
      const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d           = 2 * fe_degree - 1;
    constexpr unsigned int local_dim           = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int regular_vpatch_size = Util::pow(2, dim);
    constexpr unsigned int n_dofs_z            = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int n_dofs_per_dim  = gpu_data.n_dofs_per_dim;
    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedDataSmoother<dim, Number, SmootherVariant::GLOBAL, local_solver>
      shared_data(get_shared_data_ptr<Number>(),
                  patch_per_block,
                  n_dofs_1d,
                  local_dim);

    if (patch < gpu_data.n_patches)
      {
        unsigned int patch_type = 0;
        for (unsigned int d = 0; d < dim; ++d)
          patch_type += gpu_data.patch_type[patch * dim + d] * Util::pow(3, d);

        for (unsigned int d = 0; d < dim; ++d)
          {
            shared_data
              .local_mass[(local_patch * dim + d) * n_dofs_1d + local_tid_x] =
              gpu_data
                .eigenvalues[(patch_type * dim + d) * n_dofs_1d + local_tid_x];

            shared_data.local_laplace[((local_patch * dim + d) * n_dofs_1d +
                                       threadIdx.y) *
                                        n_dofs_1d +
                                      local_tid_x] =
              gpu_data.eigenvectors[((patch_type * dim + d) * n_dofs_1d +
                                     threadIdx.y) *
                                      n_dofs_1d +
                                    local_tid_x];
          }

        // #pragma unroll
        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x;

            const unsigned int global_dof_indices =
              (z + dim - 2) * n_dofs_per_dim * n_dofs_per_dim +
              (threadIdx.y + 1) * n_dofs_per_dim + local_tid_x + 1 +
              gpu_data.first_dof[patch];

            shared_data.local_src[index] = src[global_dof_indices];

#if SCHWARZTYPE == 2
            shared_data.local_dst[index] = 0;
#else
            shared_data.local_dst[index] = dst[global_dof_indices];
#endif
          }

        evaluate_smooth_global<dim, fe_degree, Number, local_solver>(
          local_patch, &shared_data);

        // #pragma unroll
        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x;

            const unsigned int global_dof_indices =
              (z + dim - 2) * n_dofs_per_dim * n_dofs_per_dim +
              (threadIdx.y + 1) * n_dofs_per_dim + local_tid_x + 1 +
              gpu_data.first_dof[patch];

#if SCHWARZTYPE == 2
            atomicAdd(&dst[global_dof_indices],
                      shared_data.local_dst[index] * gpu_data.relaxation);
#else
            dst[global_dof_indices] =
              shared_data.local_dst[index] * gpu_data.relaxation;
#endif
          }
      }
  }


  template <int dim,
            int fe_degree,
            typename Number,
            LaplaceVariant     lapalace,
            LocalSolverVariant local_solver>
  __global__
    typename std::enable_if<local_solver == LocalSolverVariant::Exact>::type
    loop_kernel_fused_cf(
      const Number                                                 *src,
      Number                                                       *dst,
      const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d   = 2 * fe_degree + 1;
    constexpr unsigned int n_dofs_1d_i = 2 * fe_degree - 1;
    constexpr unsigned int local_dim   = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int n_dofs_z    = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int n_dofs_per_dim  = gpu_data.n_dofs_per_dim;
    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedDataSmoother<dim, Number, SmootherVariant::ConflictFree, local_solver>
      shared_data(get_shared_data_ptr<Number>(),
                  patch_per_block,
                  n_dofs_1d,
                  local_dim);

    if (patch < gpu_data.n_patches)
      {
        shared_data.local_mass[local_patch * n_dofs_1d * n_dofs_1d +
                               threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.smooth_mass_1d[2 * n_dofs_1d * n_dofs_1d +
                                  threadIdx.y * n_dofs_1d + local_tid_x];
        shared_data.local_laplace[local_patch * n_dofs_1d * n_dofs_1d +
                                  threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.smooth_stiff_1d[2 * n_dofs_1d * n_dofs_1d +
                                   threadIdx.y * n_dofs_1d + local_tid_x];

        for (unsigned int d = 0; d < dim; ++d)
          {
            shared_data
              .local_bilaplace[(local_patch * dim + d) * n_dofs_1d * n_dofs_1d +
                               threadIdx.y * n_dofs_1d + local_tid_x] =
              gpu_data
                .smooth_bilaplace_1d[gpu_data.patch_type[patch * dim + d] *
                                       n_dofs_1d * n_dofs_1d +
                                     threadIdx.y * n_dofs_1d + local_tid_x];
          }

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x;

            const unsigned int global_dof_indices =
              z * n_dofs_per_dim * n_dofs_per_dim +
              threadIdx.y * n_dofs_per_dim + local_tid_x +
              gpu_data.first_dof[patch];

            shared_data.local_src[index] = src[global_dof_indices];

            shared_data.local_dst[index] = dst[global_dof_indices];
          }

        evaluate_smooth_cf<dim, fe_degree, Number, lapalace, local_solver>(
          local_patch, patch, &shared_data, &gpu_data);

        constexpr unsigned int local_dim_i = Util::pow(n_dofs_1d_i, dim);
        constexpr unsigned int matrix_size = Util::pow(local_dim_i, 2);
        unsigned int           patch_type  = 0;
        for (unsigned int d = 0; d < dim; ++d)
          patch_type += gpu_data.patch_type[patch * dim + d] * Util::pow(3, d);

        unsigned int linear_tid = local_tid_x + threadIdx.y * n_dofs_1d;

        if (linear_tid < n_dofs_1d_i * n_dofs_1d_i)
          for (unsigned int row = 0; row < local_dim_i; ++row)
            {
              unsigned int layer = row / (n_dofs_1d_i * n_dofs_1d_i);
              for (unsigned int z = 0; z < n_dofs_z; ++z)
                {
                  auto val =
                    gpu_data
                      .eigenvalues[patch_type * matrix_size +
                                   row * local_dim_i +
                                   z * n_dofs_1d_i * n_dofs_1d_i + linear_tid] *
                    shared_data.tmp[2 * local_patch * local_dim + local_dim +
                                    z * n_dofs_1d * n_dofs_1d + linear_tid];

                  atomicAdd(&shared_data.tmp[2 * local_patch * local_dim + row +
                                             layer * (4 * fe_degree * 2)],
                            val);
                }
            }
        __syncthreads();

        if (dim == 2)
          {
            if (linear_tid < (n_dofs_1d_i) * (n_dofs_1d_i))
              {
                int row = linear_tid / (n_dofs_1d_i) + 1;
                int col = linear_tid % (n_dofs_1d_i) + 1;

                const unsigned int index = 2 * local_patch * local_dim +
                                           (row - 1) * (n_dofs_1d_i) + col - 1;

                const unsigned int global_dof_indices =
                  row * n_dofs_per_dim + col + gpu_data.first_dof[patch];

                dst[global_dof_indices] =
                  shared_data.tmp[index] * gpu_data.relaxation;
              }
          }
        else if (dim == 3)
          {
            for (unsigned int z = 0; z < n_dofs_1d_i; ++z)
              {
                unsigned int linear_tid = local_tid_x + threadIdx.y * n_dofs_1d;

                if (linear_tid < (n_dofs_1d_i) * (n_dofs_1d_i))
                  {
                    unsigned int row = linear_tid / (n_dofs_1d_i) + 1;
                    unsigned int col = linear_tid % (n_dofs_1d_i) + 1;

                    unsigned int index = 2 * local_patch * local_dim +
                                         z * n_dofs_1d * n_dofs_1d +
                                         (row - 1) * (n_dofs_1d_i) + col - 1;

                    const unsigned int global_dof_indices =
                      (z + 1) * n_dofs_per_dim * n_dofs_per_dim +
                      row * n_dofs_per_dim + col + gpu_data.first_dof[patch];

                    dst[global_dof_indices] =
                      shared_data.tmp[index] * gpu_data.relaxation;
                  }
              }
          }
      }
  }

  template <int dim,
            int fe_degree,
            typename Number,
            LaplaceVariant     lapalace,
            LocalSolverVariant local_solver>
  __global__
    typename std::enable_if<local_solver != LocalSolverVariant::Exact>::type
    loop_kernel_fused_cf(
      const Number                                                 *src,
      Number                                                       *dst,
      const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d = 2 * fe_degree + 1;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int n_dofs_per_dim  = gpu_data.n_dofs_per_dim;
    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedDataSmoother<dim, Number, SmootherVariant::ConflictFree, local_solver>
      shared_data(get_shared_data_ptr<Number>(),
                  patch_per_block,
                  n_dofs_1d,
                  local_dim);

    if (patch < gpu_data.n_patches)
      {
        shared_data.local_mass[local_patch * n_dofs_1d * n_dofs_1d +
                               threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.smooth_mass_1d[2 * n_dofs_1d * n_dofs_1d +
                                  threadIdx.y * n_dofs_1d + local_tid_x];
        shared_data.local_laplace[local_patch * n_dofs_1d * n_dofs_1d +
                                  threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.smooth_stiff_1d[2 * n_dofs_1d * n_dofs_1d +
                                   threadIdx.y * n_dofs_1d + local_tid_x];

        for (unsigned int d = 0; d < dim; ++d)
          {
            shared_data
              .local_bilaplace[(local_patch * dim + d) * n_dofs_1d * n_dofs_1d +
                               threadIdx.y * n_dofs_1d + local_tid_x] =
              gpu_data
                .smooth_bilaplace_1d[gpu_data.patch_type[patch * dim + d] *
                                       n_dofs_1d * n_dofs_1d +
                                     threadIdx.y * n_dofs_1d + local_tid_x];
          }

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x;

            const unsigned int global_dof_indices =
              z * n_dofs_per_dim * n_dofs_per_dim +
              threadIdx.y * n_dofs_per_dim + local_tid_x +
              gpu_data.first_dof[patch];

            shared_data.local_src[index] = src[global_dof_indices];

            shared_data.local_dst[index] = dst[global_dof_indices];
          }

        evaluate_smooth_cf<dim, fe_degree, Number, lapalace, local_solver>(
          local_patch, patch, &shared_data, &gpu_data);

        if (dim == 2)
          {
            unsigned int linear_tid = local_tid_x + threadIdx.y * n_dofs_1d;

            if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
              {
                int row = linear_tid / (n_dofs_1d - 2) + 1;
                int col = linear_tid % (n_dofs_1d - 2) + 1;

                const unsigned int index = 2 * local_patch * local_dim +
                                           (row - 1) * (n_dofs_1d - 2) + col -
                                           1;

                const unsigned int global_dof_indices =
                  row * n_dofs_per_dim + col + gpu_data.first_dof[patch];

                dst[global_dof_indices] =
                  shared_data.tmp[index] * gpu_data.relaxation;
              }
          }
        else if (dim == 3)
          {
            unsigned int linear_tid = local_tid_x + threadIdx.y * n_dofs_1d;

            if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
              {
                for (unsigned int z = 0; z < n_dofs_1d - 2; ++z)
                  {
                    unsigned int row = linear_tid / (n_dofs_1d - 2) + 1;
                    unsigned int col = linear_tid % (n_dofs_1d - 2) + 1;

                    unsigned int index = 2 * local_patch * local_dim +
                                         z * n_dofs_1d * n_dofs_1d +
                                         (row - 1) * (n_dofs_1d - 2) + col - 1;

                    const unsigned int global_dof_indices =
                      (z + 1) * n_dofs_per_dim * n_dofs_per_dim +
                      row * n_dofs_per_dim + col + gpu_data.first_dof[patch];

                    dst[global_dof_indices] =
                      shared_data.tmp[index] * gpu_data.relaxation;
                  }
              }
          }
      }
  }

} // namespace PSMF

#endif // LOOP_KERNEL_CUH
