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

#include "patch_base.cuh"

namespace PSMF
{

  template <int dim,
            int fe_degree,
            typename Number,
            SmootherVariant kernel,
            typename Functor,
            DoFLayout dof_layout>
  __global__ void
  loop_kernel_seperate(
    Functor                                           func,
    const Number                                     *src,
    Number                                           *dst,
    Number                                           *tmp,
    const typename LevelVertexPatch<dim,
                                    fe_degree,
                                    Number,
                                    kernel,
                                    dof_layout>::Data gpu_data)
  {
    extern __shared__ Number data[];

    constexpr unsigned int n_dofs_1d = Functor::n_dofs_1d;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, kernel> shared_data(data,
                                                   patch_per_block,
                                                   n_dofs_1d,
                                                   local_dim);

    if (patch < gpu_data.n_patches)
      {
        shared_data.local_mass[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.global_mass_1d[threadIdx.y * n_dofs_1d + local_tid_x];
        shared_data.local_derivative[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.global_derivative_1d[threadIdx.y * n_dofs_1d + local_tid_x];

        // #pragma unroll
        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            unsigned int index = local_patch * local_dim +
                                 z * n_dofs_1d * n_dofs_1d +
                                 threadIdx.y * n_dofs_1d + local_tid_x;

            unsigned int global_dof_indices =
              z * func.get_ndofs() * func.get_ndofs() +
              threadIdx.y * func.get_ndofs() + local_tid_x +
              gpu_data.first_dof[patch];

            shared_data.local_src[index] = src[global_dof_indices];

            shared_data.local_dst[index] = dst[global_dof_indices];
          }

        func(local_patch, &gpu_data, &shared_data);

        // #pragma unroll
        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            unsigned int index = local_patch * local_dim +
                                 z * n_dofs_1d * n_dofs_1d +
                                 threadIdx.y * n_dofs_1d + local_tid_x;

            unsigned int global_dof_indices =
              z * func.get_ndofs() * func.get_ndofs() +
              threadIdx.y * func.get_ndofs() + local_tid_x +
              gpu_data.first_dof[patch];

            tmp[global_dof_indices] = shared_data.local_src[index];
          }
      }
  }

  template <int dim,
            int fe_degree,
            typename Number,
            SmootherVariant kernel,
            typename Functor,
            DoFLayout dof_layout>
  __global__ void
  loop_kernel_seperate_inv(
    Functor                                           func,
    const Number                                     *src,
    Number                                           *dst,
    const typename LevelVertexPatch<dim,
                                    fe_degree,
                                    Number,
                                    kernel,
                                    dof_layout>::Data gpu_data)
  {
    extern __shared__ Number data[];

    constexpr unsigned int n_dofs_1d = Functor::n_dofs_1d;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, kernel> shared_data(data,
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
            unsigned int index = local_patch * local_dim +
                                 z * n_dofs_1d * n_dofs_1d +
                                 threadIdx.y * n_dofs_1d + local_tid_x;

            unsigned int global_dof_indices =
              (z + 1) * func.get_ndofs() * func.get_ndofs() +
              (threadIdx.y + 1) * func.get_ndofs() + local_tid_x + 1 +
              gpu_data.first_dof[patch];

            shared_data.local_src[index] = src[global_dof_indices];

            shared_data.local_dst[index] = dst[global_dof_indices];
          }

        func(local_patch, &gpu_data, &shared_data);

        // #pragma unroll
        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            unsigned int index = local_patch * local_dim +
                                 z * n_dofs_1d * n_dofs_1d +
                                 threadIdx.y * n_dofs_1d + local_tid_x;

            unsigned int global_dof_indices =
              (z + 1) * func.get_ndofs() * func.get_ndofs() +
              (threadIdx.y + 1) * func.get_ndofs() + local_tid_x + 1 +
              gpu_data.first_dof[patch];

            dst[global_dof_indices] =
              shared_data.local_dst[index] * gpu_data.relaxation;
          }
      }
  }

  template <int dim,
            int fe_degree,
            typename Number,
            SmootherVariant kernel,
            typename Functor,
            DoFLayout dof_layout>
  __global__ void
  loop_kernel_fused(Functor                                           func,
                    const Number                                     *src,
                    Number                                           *dst,
                    const typename LevelVertexPatch<dim,
                                                    fe_degree,
                                                    Number,
                                                    kernel,
                                                    dof_layout>::Data gpu_data)
  {
    extern __shared__ Number data[];

    constexpr unsigned int n_dofs_1d = Functor::n_dofs_1d;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, kernel> shared_data(data,
                                                   patch_per_block,
                                                   n_dofs_1d,
                                                   local_dim);

    if (patch < gpu_data.n_patches)
      {
        shared_data.local_mass[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.global_mass_1d[threadIdx.y * n_dofs_1d + local_tid_x];
        shared_data.local_derivative[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.global_derivative_1d[threadIdx.y * n_dofs_1d + local_tid_x];

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            unsigned int index = local_patch * local_dim +
                                 z * n_dofs_1d * n_dofs_1d +
                                 threadIdx.y * n_dofs_1d + local_tid_x;

            unsigned int global_dof_indices =
              z * func.get_ndofs() * func.get_ndofs() +
              threadIdx.y * func.get_ndofs() + local_tid_x +
              gpu_data.first_dof[patch];

            shared_data.local_src[index] = src[global_dof_indices];

            shared_data.local_dst[index] = dst[global_dof_indices];
          }

        func(local_patch, &gpu_data, &shared_data);

        if (dim == 2)
          {
            unsigned int linear_tid = local_tid_x + threadIdx.y * n_dofs_1d;

            if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
              {
                int row = linear_tid / (n_dofs_1d - 2) + 1;
                int col = linear_tid % (n_dofs_1d - 2) + 1;

                unsigned int index =
                  local_patch * local_dim + row * n_dofs_1d + col;

                unsigned int global_dof_indices =
                  row * func.get_ndofs() + col + gpu_data.first_dof[patch];

                dst[global_dof_indices] =
                  shared_data.local_dst[index] * gpu_data.relaxation;
              }
          }
        else if (dim == 3)
          {
            for (unsigned int z = 1; z < n_dofs_1d - 1; ++z)
              {
                unsigned int linear_tid = local_tid_x + threadIdx.y * n_dofs_1d;

                if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
                  {
                    int row = linear_tid / (n_dofs_1d - 2) + 1;
                    int col = linear_tid % (n_dofs_1d - 2) + 1;

                    unsigned int index = local_patch * local_dim +
                                         z * n_dofs_1d * n_dofs_1d +
                                         row * n_dofs_1d + col;

                    unsigned int global_dof_indices =
                      z * func.get_ndofs() * func.get_ndofs() +
                      row * func.get_ndofs() + col + gpu_data.first_dof[patch];

                    dst[global_dof_indices] =
                      shared_data.local_dst[index] * gpu_data.relaxation;
                  }
              }
          }
      }
  }

} // namespace PSMF

#endif // LOOP_KERNEL_CUH