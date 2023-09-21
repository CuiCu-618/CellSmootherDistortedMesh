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


  template <int dim,
            int fe_degree,
            typename Number,
            LaplaceVariant laplace,
            bool           is_ghost = false>
  __global__ void
  laplace_kernel_basic(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, true> shared_data(get_shared_data_ptr<Number>(),
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
              .local_derivative[(local_patch * dim + d) * n_dofs_1d *
                                  n_dofs_1d +
                                threadIdx.y * n_dofs_1d + local_tid_x] =
              gpu_data.laplace_stiff_1d[gpu_data.patch_type[patch * dim + d] *
                                          n_dofs_1d * n_dofs_1d +
                                        threadIdx.y * n_dofs_1d + local_tid_x];
          }

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index =
              z * n_dofs_1d * n_dofs_1d + threadIdx.y * n_dofs_1d + local_tid_x;

            unsigned int global_dof_indices;

            if constexpr (is_ghost)
              global_dof_indices =
                gpu_data.patch_dofs[patch * local_dim + gpu_data.h_to_l[index]];
            else
              {
                const unsigned int global_index =
                  Util::compute_indices<dim, fe_degree>(
                    &gpu_data.first_dof[patch * (1 << dim)],
                    local_patch,
                    local_tid_x,
                    threadIdx.y,
                    z);
                global_dof_indices = gpu_data.global_to_local(global_index);
              }

            shared_data.local_src[local_patch * local_dim + index] =
              src[global_dof_indices];

            shared_data.local_dst[local_patch * local_dim + index] = 0.;
          }

        evaluate_laplace<dim, fe_degree, Number, laplace>(local_patch,
                                                          &shared_data);

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index =
              z * n_dofs_1d * n_dofs_1d + threadIdx.y * n_dofs_1d + local_tid_x;

            unsigned int global_dof_indices;

            if constexpr (is_ghost)
              global_dof_indices =
                gpu_data.patch_dofs[patch * local_dim + gpu_data.h_to_l[index]];
            else
              {
                const unsigned int global_index =
                  Util::compute_indices<dim, fe_degree>(
                    &gpu_data.first_dof[patch * (1 << dim)],
                    local_patch,
                    local_tid_x,
                    threadIdx.y,
                    z);
                global_dof_indices = gpu_data.global_to_local(global_index);
              }

            atomicAdd(&dst[global_dof_indices],
                      shared_data.local_dst[local_patch * local_dim + index]);
          }
      }
  }

  // Load patch data cell by cell
  template <int dim, int fe_degree, typename Number, LaplaceVariant laplace>
  __global__ void
  laplace_kernel_basic_cell(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, true> shared_data(get_shared_data_ptr<Number>(),
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
              .local_derivative[(local_patch * dim + d) * n_dofs_1d *
                                  n_dofs_1d +
                                threadIdx.y * n_dofs_1d + local_tid_x] =
              gpu_data.laplace_stiff_1d[gpu_data.patch_type[patch * dim + d] *
                                          n_dofs_1d * n_dofs_1d +
                                        threadIdx.y * n_dofs_1d + local_tid_x];
          }

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int linear_tid =
              z * n_dofs_1d * n_dofs_1d + threadIdx.y * n_dofs_1d + local_tid_x;
            const unsigned int index =
              local_patch * local_dim + gpu_data.l_to_h[linear_tid];

            const unsigned int global_index =
              Util::compute_indices_cell<dim, fe_degree>(
                &gpu_data.first_dof[patch * (1 << dim)], linear_tid);

            const unsigned int global_dof_indices =
              gpu_data.global_to_local(global_index);

            shared_data.local_src[index] = src[global_dof_indices];

            shared_data.local_dst[index] = 0.;
          }

        evaluate_laplace<dim, fe_degree, Number, laplace>(local_patch,
                                                          &shared_data);

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int linear_tid =
              z * n_dofs_1d * n_dofs_1d + threadIdx.y * n_dofs_1d + local_tid_x;
            const unsigned int index =
              local_patch * local_dim + gpu_data.l_to_h[linear_tid];

            const unsigned int global_index =
              Util::compute_indices_cell<dim, fe_degree>(
                &gpu_data.first_dof[patch * (1 << dim)], linear_tid);

            const unsigned int global_dof_indices =
              gpu_data.global_to_local(global_index);

            atomicAdd(&dst[global_dof_indices], shared_data.local_dst[index]);
          }
      }
  }

  template <int dim, int fe_degree, typename Number, LaplaceVariant laplace>
  __global__ void
  laplace_kernel_tensorcore(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d         = 2 * fe_degree + 2;
    constexpr unsigned int n_dofs_1d_padding = n_dofs_1d + Util::padding;
    constexpr unsigned int local_dim         = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int local_dim_padding =
      Util::pow(n_dofs_1d, dim - 1) * n_dofs_1d_padding;
    constexpr unsigned int n_dofs_z = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, true> shared_data(get_shared_data_ptr<Number>(),
                                                 patch_per_block,
                                                 n_dofs_1d,
                                                 local_dim_padding,
                                                 n_dofs_1d_padding);

    if (patch < gpu_data.n_patches)
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            shared_data
              .local_mass[(local_patch * dim + d) * n_dofs_1d *
                            n_dofs_1d_padding +
                          threadIdx.y * n_dofs_1d_padding + local_tid_x] =
              gpu_data.laplace_mass_1d[gpu_data.patch_type[patch * dim + d] *
                                         n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x];

            shared_data
              .local_derivative[(local_patch * dim + d) * n_dofs_1d *
                                  n_dofs_1d_padding +
                                threadIdx.y * n_dofs_1d_padding + local_tid_x] =
              gpu_data.laplace_stiff_1d[gpu_data.patch_type[patch * dim + d] *
                                          n_dofs_1d * n_dofs_1d +
                                        threadIdx.y * n_dofs_1d + local_tid_x];
          }

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index_padding =
              local_patch * local_dim_padding +
              z * n_dofs_1d * n_dofs_1d_padding +
              threadIdx.y * n_dofs_1d_padding + local_tid_x;

            const unsigned int global_index =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * (1 << dim)],
                local_patch,
                local_tid_x,
                threadIdx.y,
                z);

            const unsigned int global_dof_indices =
              gpu_data.global_to_local(global_index);

            shared_data.local_src[index_padding] = src[global_dof_indices];

            shared_data.local_dst[index_padding] = 0.;
          }

        evaluate_laplace<dim, fe_degree, Number, laplace>(local_patch,
                                                          &shared_data);

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index_padding =
              local_patch * local_dim_padding +
              z * n_dofs_1d * n_dofs_1d_padding +
              threadIdx.y * n_dofs_1d_padding + local_tid_x;

            const unsigned int global_index =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * (1 << dim)],
                local_patch,
                local_tid_x,
                threadIdx.y,
                z);

            const unsigned int global_dof_indices =
              gpu_data.global_to_local(global_index);

            atomicAdd(&dst[global_dof_indices],
                      shared_data.local_dst[index_padding]);
          }
      }
  }

  template <int dim, int fe_degree, typename Number, LaplaceVariant laplace>
  __global__ void
  laplace_kernel_tensorcore_mma(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d         = 2 * fe_degree + 2;
    constexpr unsigned int n_dofs_1d_padding = n_dofs_1d + Util::padding;
    constexpr unsigned int local_dim         = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int local_dim_padding =
      Util::pow(n_dofs_1d, dim - 1) * n_dofs_1d_padding;
    constexpr unsigned int n_dofs_z = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, true> shared_data(get_shared_data_ptr<Number>(),
                                                 patch_per_block,
                                                 n_dofs_1d,
                                                 local_dim_padding,
                                                 n_dofs_1d_padding);

    if (patch < gpu_data.n_patches)
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            shared_data
              .local_mass[(local_patch * dim + d) * n_dofs_1d *
                              n_dofs_1d_padding +
                            threadIdx.y * n_dofs_1d_padding + local_tid_x ^
                          (Util::get_permute_base<fe_degree>(threadIdx.y, 0))] =
              gpu_data.laplace_mass_1d[gpu_data.patch_type[patch * dim + d] *
                                         n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x];

            shared_data.local_derivative
              [(local_patch * dim + d) * n_dofs_1d * n_dofs_1d_padding +
                 threadIdx.y * n_dofs_1d_padding + local_tid_x ^
               (Util::get_permute_base<fe_degree>(threadIdx.y, 0))] =
              gpu_data.laplace_stiff_1d[gpu_data.patch_type[patch * dim + d] *
                                          n_dofs_1d * n_dofs_1d +
                                        threadIdx.y * n_dofs_1d + local_tid_x];
          }

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index_padding =
              local_patch * local_dim_padding +
                z * n_dofs_1d * n_dofs_1d_padding +
                threadIdx.y * n_dofs_1d_padding + local_tid_x ^
              (Util::get_permute_base<fe_degree>(threadIdx.y, z));

            const unsigned int global_index =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * (1 << dim)],
                local_patch,
                local_tid_x,
                threadIdx.y,
                z);

            const unsigned int global_dof_indices =
              gpu_data.global_to_local(global_index);

            shared_data.local_src[index_padding] = src[global_dof_indices];

            shared_data.local_dst[index_padding] = 0.;
          }

        evaluate_laplace<dim, fe_degree, Number, laplace>(local_patch,
                                                          &shared_data);

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index_padding =
              local_patch * local_dim_padding +
                z * n_dofs_1d * n_dofs_1d_padding +
                threadIdx.y * n_dofs_1d_padding + local_tid_x ^
              (Util::get_permute_base<fe_degree>(threadIdx.y, z));

            const unsigned int global_index =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * (1 << dim)],
                local_patch,
                local_tid_x,
                threadIdx.y,
                z);

            const unsigned int global_dof_indices =
              gpu_data.global_to_local(global_index);

            atomicAdd(&dst[global_dof_indices],
                      shared_data.local_dst[index_padding]);
          }
      }
  }


  template <int dim, int fe_degree, typename Number>
  __global__ void
  loop_kernel_seperate_inv(
    const Number                                                 *src,
    Number                                                       *dst,
    Number                                                       *solution,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d           = 2 * fe_degree;
    constexpr unsigned int local_dim           = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int regular_vpatch_size = Util::pow(2, dim);
    constexpr unsigned int n_dofs_z            = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, false> shared_data(get_shared_data_ptr<Number>(),
                                                  patch_per_block,
                                                  n_dofs_1d,
                                                  local_dim);

    if (patch < gpu_data.n_patches)
      {
        shared_data.local_mass[local_tid_x] = gpu_data.eigenvalues[local_tid_x];
        shared_data.local_derivative[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.eigenvectors[threadIdx.y * n_dofs_1d + local_tid_x];

        // #pragma unroll
        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x;

            const unsigned int global_index =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * regular_vpatch_size],
                local_patch,
                local_tid_x + 1,
                threadIdx.y + 1,
                z + dim - 2);

            const unsigned int global_dof_indices =
              gpu_data.global_to_local(global_index);

            shared_data.local_src[index] = src[global_dof_indices];

            shared_data.local_dst[index] = dst[global_dof_indices];
          }

        evaluate_smooth_inv<dim, fe_degree, Number, SmootherVariant::GLOBAL>(
          local_patch, &shared_data);

        // #pragma unroll
        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x;

            const unsigned int global_index =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * regular_vpatch_size],
                local_patch,
                local_tid_x + 1,
                threadIdx.y + 1,
                z + dim - 2);

            const unsigned int global_dof_indices =
              gpu_data.global_to_local(global_index);

            solution[global_dof_indices] =
              shared_data.local_dst[index] * gpu_data.relaxation;
          }
      }
  }

  template <int dim,
            int fe_degree,
            typename Number,
            LaplaceVariant lapalace,
            bool           is_ghost = false>
  __global__ void
  loop_kernel_fused_l(
    const Number                                                 *src,
    Number                                                       *dst,
    Number                                                       *solution,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d           = 2 * fe_degree + 2;
    constexpr unsigned int local_dim           = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int regular_vpatch_size = Util::pow(2, dim);
    constexpr unsigned int n_dofs_z            = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, false> shared_data(get_shared_data_ptr<Number>(),
                                                  patch_per_block,
                                                  n_dofs_1d,
                                                  local_dim);

    if (patch < gpu_data.n_patches)
      {
        shared_data.local_mass[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.smooth_mass_1d[threadIdx.y * n_dofs_1d + local_tid_x];
        shared_data.local_derivative[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.smooth_stiff_1d[threadIdx.y * n_dofs_1d + local_tid_x];

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index =
              z * n_dofs_1d * n_dofs_1d + threadIdx.y * n_dofs_1d + local_tid_x;

            unsigned int global_dof_indices;

            if constexpr (is_ghost)
              global_dof_indices =
                gpu_data.patch_dofs[patch * local_dim + gpu_data.h_to_l[index]];
            else
              {
                const unsigned int global_index =
                  Util::compute_indices<dim, fe_degree>(
                    &gpu_data.first_dof[patch * (1 << dim)],
                    local_patch,
                    local_tid_x,
                    threadIdx.y,
                    z);
                global_dof_indices = gpu_data.global_to_local(global_index);
              }

            shared_data.local_src[local_patch * local_dim + index] =
              src[global_dof_indices];

            shared_data.local_dst[local_patch * local_dim + index] =
              dst[global_dof_indices];
          }

        evaluate_smooth<dim,
                        fe_degree,
                        Number,
                        lapalace,
                        SmootherVariant::FUSED_L>(local_patch,
                                                  &shared_data,
                                                  &gpu_data);

        const unsigned int linear_tid = local_tid_x + threadIdx.y * n_dofs_1d;
        if (dim == 2)
          {
            if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
              {
                int row = linear_tid / (n_dofs_1d - 2) + 1;
                int col = linear_tid % (n_dofs_1d - 2) + 1;

                const unsigned int index =
                  local_patch * local_dim + row * n_dofs_1d + col;

                const unsigned int global_index =
                  Util::compute_indices<dim, fe_degree>(
                    &gpu_data.first_dof[patch * regular_vpatch_size],
                    local_patch,
                    col,
                    row,
                    0);

                const unsigned int global_dof_indices =
                  gpu_data.global_to_local(global_index);

                solution[global_dof_indices] =
                  shared_data.local_dst[index] * gpu_data.relaxation;
              }
          }
        else if (dim == 3)
          {
            if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
              for (unsigned int z = 1; z < n_dofs_1d - 1; ++z)
                {
                  int row = linear_tid / (n_dofs_1d - 2) + 1;
                  int col = linear_tid % (n_dofs_1d - 2) + 1;

                  unsigned int index_g =
                    (z - 1) * (n_dofs_1d - 2) * (n_dofs_1d - 2) +
                    (row - 1) * (n_dofs_1d - 2) + col - 1;

                  unsigned int index =
                    z * n_dofs_1d * n_dofs_1d + row * n_dofs_1d + col;

                  unsigned int global_dof_indices;

                  if constexpr (is_ghost)
                    global_dof_indices =
                      gpu_data.patch_dofs[patch * local_dim +
                                          gpu_data.l_to_h[index_g]];
                  else
                    {
                      const unsigned int global_index =
                        Util::compute_indices<dim, fe_degree>(
                          &gpu_data.first_dof[patch * regular_vpatch_size],
                          local_patch,
                          col,
                          row,
                          z);
                      global_dof_indices =
                        gpu_data.global_to_local(global_index);
                    }

                  solution[global_dof_indices] =
                    shared_data.local_dst[local_patch * local_dim + index] *
                    gpu_data.relaxation;
                }
          }
      }
  }


  template <int dim,
            int fe_degree,
            typename Number,
            LaplaceVariant lapalace,
            bool           is_ghost = false>
  __global__ void
  loop_kernel_fused_cf(
    const Number                                                 *src,
    Number                                                       *dst,
    Number                                                       *solution,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d           = 2 * fe_degree + 2;
    constexpr unsigned int local_dim           = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int regular_vpatch_size = Util::pow(2, dim);
    constexpr unsigned int n_dofs_z            = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, false> shared_data(get_shared_data_ptr<Number>(),
                                                  patch_per_block,
                                                  n_dofs_1d,
                                                  local_dim);

    if (patch < gpu_data.n_patches)
      {
        shared_data.local_mass[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.smooth_mass_1d[threadIdx.y * n_dofs_1d + local_tid_x];
        shared_data.local_derivative[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.smooth_stiff_1d[threadIdx.y * n_dofs_1d + local_tid_x];

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index =
              z * n_dofs_1d * n_dofs_1d + threadIdx.y * n_dofs_1d + local_tid_x;

            unsigned int global_dof_indices;

            if constexpr (is_ghost)
              global_dof_indices =
                gpu_data.patch_dofs[patch * local_dim + gpu_data.h_to_l[index]];
            else
              {
                const unsigned int global_index =
                  Util::compute_indices<dim, fe_degree>(
                    &gpu_data.first_dof[patch * (1 << dim)],
                    local_patch,
                    local_tid_x,
                    threadIdx.y,
                    z);
                global_dof_indices = gpu_data.global_to_local(global_index);
              }

            shared_data.local_src[local_patch * local_dim + index] =
              src[global_dof_indices];

            shared_data.local_dst[local_patch * local_dim + index] =
              dst[global_dof_indices];
          }

        evaluate_smooth_cf<dim,
                           fe_degree,
                           Number,
                           lapalace,
                           SmootherVariant::ConflictFree>(local_patch,
                                                          &shared_data,
                                                          &gpu_data);

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

                const unsigned int global_index =
                  Util::compute_indices<dim, fe_degree>(
                    &gpu_data.first_dof[patch * regular_vpatch_size],
                    local_patch,
                    col,
                    row,
                    0);

                const unsigned int global_dof_indices =
                  gpu_data.global_to_local(global_index);

                solution[global_dof_indices] =
                  shared_data.tmp[index] * gpu_data.relaxation;
              }
          }
        else if (dim == 3)
          {
            for (unsigned int z = 0; z < n_dofs_1d - 2; ++z)
              {
                unsigned int linear_tid = local_tid_x + threadIdx.y * n_dofs_1d;

                if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
                  {
                    unsigned int row = linear_tid / (n_dofs_1d - 2);
                    unsigned int col = linear_tid % (n_dofs_1d - 2);

                    unsigned int index_g =
                      z * (n_dofs_1d - 2) * (n_dofs_1d - 2) +
                      row * (n_dofs_1d - 2) + col;
                    unsigned int index =
                      z * n_dofs_1d * n_dofs_1d + row * (n_dofs_1d - 2) + col;

                    unsigned int global_dof_indices;

                    if constexpr (is_ghost)
                      global_dof_indices =
                        gpu_data.patch_dofs[patch * local_dim +
                                            gpu_data.l_to_h[index_g]];
                    else
                      {
                        const unsigned int global_index =
                          Util::compute_indices<dim, fe_degree>(
                            &gpu_data.first_dof[patch * regular_vpatch_size],
                            local_patch,
                            col + 1,
                            row + 1,
                            z + 1);
                        global_dof_indices =
                          gpu_data.global_to_local(global_index);
                      }

                    solution[global_dof_indices] =
                      shared_data
                        .tmp[(dim - 1) * local_patch * local_dim + index] *
                      gpu_data.relaxation;
                  }
              }
          }
      }
  }

  template <int dim, int fe_degree, typename Number, LaplaceVariant lapalace>
  __global__ void
  loop_kernel_fused_exact(
    const Number                                                 *src,
    Number                                                       *dst,
    Number                                                       *solution,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d           = 2 * fe_degree + 2;
    constexpr unsigned int local_dim           = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int regular_vpatch_size = Util::pow(2, dim);
    constexpr unsigned int n_dofs_z            = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, false> shared_data(get_shared_data_ptr<Number>(),
                                                  patch_per_block,
                                                  n_dofs_1d,
                                                  local_dim);

    if (patch < gpu_data.n_patches)
      {
        shared_data.local_mass[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.smooth_mass_1d[threadIdx.y * n_dofs_1d + local_tid_x];
        shared_data.local_derivative[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.smooth_stiff_1d[threadIdx.y * n_dofs_1d + local_tid_x];

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x;

            const unsigned int global_index =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * regular_vpatch_size],
                local_patch,
                local_tid_x,
                threadIdx.y,
                z);

            const unsigned int global_dof_indices =
              gpu_data.global_to_local(global_index);

            shared_data.local_src[index] = src[global_dof_indices];

            shared_data.local_dst[index] = dst[global_dof_indices];
          }

        evaluate_smooth_exact<dim,
                              fe_degree,
                              Number,
                              lapalace,
                              SmootherVariant::ExactRes>(local_patch,
                                                         &shared_data,
                                                         &gpu_data);

        const unsigned int linear_tid = local_tid_x + threadIdx.y * n_dofs_1d;
        if (dim == 2)
          {
            if (linear_tid < (n_dofs_1d - 4) * (n_dofs_1d - 4))
              {
                int row = linear_tid / (n_dofs_1d - 4) + 2;
                int col = linear_tid % (n_dofs_1d - 4) + 2;

                const unsigned int index =
                  local_patch * local_dim + row * n_dofs_1d + col;

                const unsigned int global_index =
                  Util::compute_indices<dim, fe_degree>(
                    &gpu_data.first_dof[patch * regular_vpatch_size],
                    local_patch,
                    col,
                    row,
                    0);

                const unsigned int global_dof_indices =
                  gpu_data.global_to_local(global_index);

                solution[global_dof_indices] =
                  shared_data.local_dst[index] * gpu_data.relaxation;
              }
          }
        else if (dim == 3)
          {
            if (linear_tid < (n_dofs_1d - 4) * (n_dofs_1d - 4))
              for (unsigned int z = 2; z < n_dofs_1d - 2; ++z)
                {
                  int row = linear_tid / (n_dofs_1d - 4) + 2;
                  int col = linear_tid % (n_dofs_1d - 4) + 2;

                  unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       row * n_dofs_1d + col;

                  const unsigned int global_index =
                    Util::compute_indices<dim, fe_degree>(
                      &gpu_data.first_dof[patch * regular_vpatch_size],
                      local_patch,
                      col,
                      row,
                      z);

                  const unsigned int global_dof_indices =
                    gpu_data.global_to_local(global_index);

                  solution[global_dof_indices] =
                    shared_data.local_dst[index] * gpu_data.relaxation;
                }
          }
      }
  }

} // namespace PSMF

#endif // LOOP_KERNEL_CUH