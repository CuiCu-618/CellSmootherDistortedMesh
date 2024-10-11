/**
 * @file cell_loop_kernel.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief collection of global functions
 * @version 1.0
 * @date 2022-12-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef CELL_LOOP_KERNEL_CUH
#define CELL_LOOP_KERNEL_CUH

#include "cell_base.cuh"
#include "cuda_matrix_free.cuh"
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
  // extern __shared__ double data_d[];
  // extern __shared__ float  data_f[];
  //
  // template <typename Number>
  // __device__ inline Number *
  // get_shared_data_ptr();
  //
  // template <>
  // __device__ inline double *
  // get_shared_data_ptr()
  // {
  //   return data_d;
  // }
  //
  // template <>
  // __device__ inline float *
  // get_shared_data_ptr()
  // {
  //   return data_f;
  // }


  template <int dim, int fe_degree, typename Number, bool is_ghost = false>
  __global__ void
  cell_loop_kernel_seperate_inv(
    const Number                                               *src,
    Number                                                     *dst,
    Number                                                     *solution,
    const typename LevelCellPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d = fe_degree + 1;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int cell_per_block = gpu_data.cell_per_block;
    const unsigned int local_cell     = threadIdx.x / n_dofs_1d;
    const unsigned int cell        = local_cell + cell_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;
    const unsigned int tid         = threadIdx.y * n_dofs_1d + local_tid_x;

    CellSharedMemData<dim, Number, false> shared_data(
      get_shared_data_ptr<Number>(), cell_per_block, n_dofs_1d, local_dim);

    if (cell < gpu_data.n_cells)
      {
        unsigned int cell_type = 0;
        for (unsigned int d = 0; d < dim; ++d)
          cell_type += gpu_data.cell_type[cell * dim + d] * Util::pow(3, d);

        for (unsigned int d = 0; d < dim; ++d)
          {
            shared_data.local_mass[local_cell * n_dofs_1d * dim +
                                   d * n_dofs_1d + local_tid_x] =
              gpu_data.eigenvalues[cell_type * n_dofs_1d * dim + d * n_dofs_1d +
                                   local_tid_x];
            shared_data
              .local_derivative[local_cell * n_dofs_1d * n_dofs_1d * dim +
                                d * n_dofs_1d * n_dofs_1d + tid] =
              gpu_data.eigenvectors[cell_type * n_dofs_1d * n_dofs_1d * dim +
                                    d * n_dofs_1d * n_dofs_1d + tid];
          }

        // #pragma unroll
        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = z * n_dofs_1d * n_dofs_1d + tid;

            types::global_dof_index global_dof_indices;

            if constexpr (is_ghost)
              {
              }
            else
              {
                const unsigned int global_index =
                  gpu_data.first_dof[cell] + index;

                global_dof_indices = gpu_data.global_to_local(global_index);
              }

            shared_data.local_src[local_cell * local_dim + index] =
              src[global_dof_indices];

            shared_data.local_dst[local_cell * local_dim + index] =
              dst[global_dof_indices];
          }

        evaluate_cell_smooth_inv<dim, fe_degree, Number, SmootherVariant::MCS>(
          local_cell, &shared_data);

        // #pragma unroll
        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = z * n_dofs_1d * n_dofs_1d + tid;

            types::global_dof_index global_dof_indices;

            if constexpr (is_ghost)
              {
              }
            else
              {
                const unsigned int global_index =
                  gpu_data.first_dof[cell] + index;

                global_dof_indices = gpu_data.global_to_local(global_index);
              }

            dst[global_dof_indices] +=
              shared_data.local_dst[local_cell * local_dim + index] *
              gpu_data.relaxation;
          }
      }
  }


  template <int dim,
            int fe_degree,
            typename Number,
            SmootherVariant smooth,
            bool            is_ghost = false>
  __global__ void __launch_bounds__(256, 1) cell_loop_kernel_seperate_inv_cg(
    const Number                                               *src,
    Number                                                     *dst,
    Number                                                     *solution,
    const typename LevelCellPatch<dim, fe_degree, Number>::Data gpu_data,
    typename MatrixFree<dim, Number>::Data                      fe_data)
  {
    constexpr unsigned int n_dofs_1d = fe_degree + 1;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int cell_per_block = gpu_data.cell_per_block;
    const unsigned int local_cell     = threadIdx.x / n_dofs_1d;
    const unsigned int cell        = local_cell + cell_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;
    const unsigned int tid         = threadIdx.y * n_dofs_1d + local_tid_x;

    if (cell < gpu_data.n_cells)
      {
        CellSharedMemData<dim, Number, false, smooth> shared_data(
          get_shared_data_ptr<Number>(), cell_per_block, n_dofs_1d, local_dim);

        auto global_cell_id = gpu_data.local_cell_to_global[cell];

        unsigned int cell_type = 0;
        for (unsigned int d = 0; d < dim; ++d)
          cell_type += gpu_data.cell_type[cell * dim + d] * Util::pow(3, d);

        for (unsigned int d = 0; d < dim; ++d)
          {
            shared_data.local_mass[local_cell * n_dofs_1d * dim +
                                   d * n_dofs_1d + local_tid_x] =
              gpu_data.eigenvalues[cell_type * n_dofs_1d * dim + d * n_dofs_1d +
                                   local_tid_x];

            shared_data
              .local_derivative[local_cell * n_dofs_1d * n_dofs_1d * dim +
                                d * n_dofs_1d * n_dofs_1d + tid] =
              gpu_data.eigenvectors[cell_type * n_dofs_1d * n_dofs_1d * dim +
                                    d * n_dofs_1d * n_dofs_1d + tid];
          }

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = z * n_dofs_1d * n_dofs_1d + tid;

            types::global_dof_index global_dof_indices;

            if constexpr (is_ghost)
              {
              }
            else
              {
                const unsigned int global_index =
                  gpu_data.first_dof[cell] + index;

                global_dof_indices = gpu_data.global_to_local(global_index);
              }

            shared_data.local_src[local_cell * local_dim + index] =
              src[global_dof_indices];

            shared_data.local_dst[local_cell * local_dim + index] = 0;
            // dst[global_dof_indices];
          }

        if constexpr (smooth == SmootherVariant::MCS_CG)
          evaluate_cell_smooth_inv_cg<dim,
                                      fe_degree,
                                      Number,
                                      SmootherVariant::MCS_CG>(local_cell,
                                                               global_cell_id,
                                                               &shared_data,
                                                               &fe_data);
        else if (smooth == SmootherVariant::MCS_PCG)
          evaluate_cell_smooth_inv_pcg<dim,
                                       fe_degree,
                                       Number,
                                       SmootherVariant::MCS_PCG>(local_cell,
                                                                 global_cell_id,
                                                                 &shared_data,
                                                                 &fe_data);

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = z * n_dofs_1d * n_dofs_1d + tid;

            types::global_dof_index global_dof_indices;

            if constexpr (is_ghost)
              {
              }
            else
              {
                const unsigned int global_index =
                  gpu_data.first_dof[cell] + index;

                global_dof_indices = gpu_data.global_to_local(global_index);
              }

            dst[global_dof_indices] +=
              shared_data.local_dst[local_cell * local_dim + index] *
              gpu_data.relaxation;

            if (tid == 0)
              {
                atomicAdd(&solution[0],
                          shared_data.local_src[local_cell * local_dim + 0]);
                atomicAdd(&solution[1],
                          shared_data.local_src[local_cell * local_dim + 1]);
                atomicAdd(&solution[2],
                          shared_data.local_src[local_cell * local_dim + 2]);
              }
          }
      }
  }


  // template <int dim,
  //           int fe_degree,
  //           typename Number,
  //           LaplaceVariant lapalace,
  //           bool           is_ghost = false>
  // __global__ void
  // loop_kernel_fused_l(
  //   const Number                                                 *src,
  //   Number                                                       *dst,
  //   Number                                                       *solution,
  //   const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  // {
  //   constexpr unsigned int n_dofs_1d           = 2 * fe_degree + 2;
  //   constexpr unsigned int local_dim           = Util::pow(n_dofs_1d, dim);
  //   constexpr unsigned int regular_vpatch_size = Util::pow(2, dim);
  //   constexpr unsigned int n_dofs_z            = dim == 2 ? 1 : n_dofs_1d;
  //
  //   const unsigned int cell_per_block = gpu_data.cell_per_block;
  //   const unsigned int local_cell     = threadIdx.x / n_dofs_1d;
  //   const unsigned int cell       = local_cell + cell_per_block *
  //   blockIdx.x; const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;
  //
  //   SharedMemData<dim, Number, false>
  //   shared_data(get_shared_data_ptr<Number>(),
  //                                                 cell_per_block,
  //                                                 n_dofs_1d,
  //                                                 local_dim);
  //
  //   if (cell < gpu_data.n_cells)
  //     {
  //       shared_data.local_mass[threadIdx.y * n_dofs_1d + local_tid_x] =
  //         gpu_data.smooth_mass_1d[threadIdx.y * n_dofs_1d + local_tid_x];
  //       shared_data.local_derivative[threadIdx.y * n_dofs_1d + local_tid_x] =
  //         gpu_data.smooth_stiff_1d[threadIdx.y * n_dofs_1d + local_tid_x];
  //
  //       for (unsigned int z = 0; z < n_dofs_z; ++z)
  //         {
  //           const unsigned int index =
  //             z * n_dofs_1d * n_dofs_1d + threadIdx.y * n_dofs_1d +
  //             local_tid_x;
  //
  //           types::global_dof_index global_dof_indices;
  //
  //           if constexpr (is_ghost)
  //             global_dof_indices =
  //               gpu_data.patch_dofs[cell * local_dim +
  //               gpu_data.h_to_l[index]];
  //           else
  //             {
  //               const types::global_dof_index global_index =
  //                 Util::compute_indices<dim, fe_degree>(
  //                   &gpu_data.first_dof[cell * (1 << dim)],
  //                   local_cell,
  //                   local_tid_x,
  //                   threadIdx.y,
  //                   z);
  //               global_dof_indices = gpu_data.global_to_local(global_index);
  //             }
  //
  //           shared_data.local_src[local_cell * local_dim + index] =
  //             src[global_dof_indices];
  //
  //           shared_data.local_dst[local_cell * local_dim + index] =
  //             dst[global_dof_indices];
  //         }
  //
  //       evaluate_smooth<dim,
  //                       fe_degree,
  //                       Number,
  //                       lapalace,
  //                       SmootherVariant::FUSED_L>(local_cell,
  //                                                 &shared_data,
  //                                                 &gpu_data);
  //
  //       const unsigned int linear_tid = local_tid_x + threadIdx.y *
  //       n_dofs_1d; if (dim == 2)
  //         {
  //           if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
  //             {
  //               int row = linear_tid / (n_dofs_1d - 2) + 1;
  //               int col = linear_tid % (n_dofs_1d - 2) + 1;
  //
  //               unsigned int index_g = (row - 1) * (n_dofs_1d - 2) + col - 1;
  //
  //               unsigned int index = row * n_dofs_1d + col;
  //
  //               types::global_dof_index global_dof_indices;
  //
  //               if constexpr (is_ghost)
  //                 global_dof_indices =
  //                   gpu_data
  //                     .patch_dofs[cell * local_dim +
  //                     gpu_data.l_to_h[index_g]];
  //               else
  //                 {
  //                   const types::global_dof_index global_index =
  //                     Util::compute_indices<dim, fe_degree>(
  //                       &gpu_data.first_dof[cell * regular_vpatch_size],
  //                       local_cell,
  //                       col,
  //                       row,
  //                       0);
  //                   global_dof_indices =
  //                   gpu_data.global_to_local(global_index);
  //                 }
  //
  //               solution[global_dof_indices] =
  //                 shared_data.local_dst[local_cell * local_dim + index] *
  //                 gpu_data.relaxation;
  //             }
  //         }
  //       else if (dim == 3)
  //         {
  //           if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
  //             for (unsigned int z = 1; z < n_dofs_1d - 1; ++z)
  //               {
  //                 int row = linear_tid / (n_dofs_1d - 2) + 1;
  //                 int col = linear_tid % (n_dofs_1d - 2) + 1;
  //
  //                 unsigned int index_g =
  //                   (z - 1) * (n_dofs_1d - 2) * (n_dofs_1d - 2) +
  //                   (row - 1) * (n_dofs_1d - 2) + col - 1;
  //
  //                 unsigned int index =
  //                   z * n_dofs_1d * n_dofs_1d + row * n_dofs_1d + col;
  //
  //                 types::global_dof_index global_dof_indices;
  //
  //                 if constexpr (is_ghost)
  //                   global_dof_indices =
  //                     gpu_data.patch_dofs[cell * local_dim +
  //                                         gpu_data.l_to_h[index_g]];
  //                 else
  //                   {
  //                     const types::global_dof_index global_index =
  //                       Util::compute_indices<dim, fe_degree>(
  //                         &gpu_data.first_dof[cell * regular_vpatch_size],
  //                         local_cell,
  //                         col,
  //                         row,
  //                         z);
  //                     global_dof_indices =
  //                       gpu_data.global_to_local(global_index);
  //                   }
  //
  //                 solution[global_dof_indices] =
  //                   shared_data.local_dst[local_cell * local_dim + index] *
  //                   gpu_data.relaxation;
  //               }
  //         }
  //     }
  // }
  //
  //
  // template <int dim,
  //           int fe_degree,
  //           typename Number,
  //           LaplaceVariant lapalace,
  //           bool           is_ghost = false>
  // __global__ void
  // loop_kernel_fused_cf(
  //   const Number                                                 *src,
  //   Number                                                       *dst,
  //   Number                                                       *solution,
  //   const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  // {
  //   constexpr unsigned int n_dofs_1d           = 2 * fe_degree + 2;
  //   constexpr unsigned int local_dim           = Util::pow(n_dofs_1d, dim);
  //   constexpr unsigned int regular_vpatch_size = Util::pow(2, dim);
  //   constexpr unsigned int n_dofs_z            = dim == 2 ? 1 : n_dofs_1d;
  //
  //   const unsigned int cell_per_block = gpu_data.cell_per_block;
  //   const unsigned int local_cell     = threadIdx.x / n_dofs_1d;
  //   const unsigned int cell       = local_cell + cell_per_block *
  //   blockIdx.x; const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;
  //
  //   SharedMemData<dim, Number, false>
  //   shared_data(get_shared_data_ptr<Number>(),
  //                                                 cell_per_block,
  //                                                 n_dofs_1d,
  //                                                 local_dim);
  //
  //   if (cell < gpu_data.n_cells)
  //     {
  //       shared_data.local_mass[threadIdx.y * n_dofs_1d + local_tid_x] =
  //         gpu_data.smooth_mass_1d[threadIdx.y * n_dofs_1d + local_tid_x];
  //       shared_data.local_derivative[threadIdx.y * n_dofs_1d + local_tid_x] =
  //         gpu_data.smooth_stiff_1d[threadIdx.y * n_dofs_1d + local_tid_x];
  //
  //       for (unsigned int z = 0; z < n_dofs_z; ++z)
  //         {
  //           const unsigned int index =
  //             z * n_dofs_1d * n_dofs_1d + threadIdx.y * n_dofs_1d +
  //             local_tid_x;
  //
  //           types::global_dof_index global_dof_indices;
  //
  //           if constexpr (is_ghost)
  //             global_dof_indices =
  //               gpu_data.patch_dofs[cell * local_dim +
  //               gpu_data.h_to_l[index]];
  //           else
  //             {
  //               const types::global_dof_index global_index =
  //                 Util::compute_indices<dim, fe_degree>(
  //                   &gpu_data.first_dof[cell * (1 << dim)],
  //                   local_cell,
  //                   local_tid_x,
  //                   threadIdx.y,
  //                   z);
  //               global_dof_indices = gpu_data.global_to_local(global_index);
  //             }
  //
  //           shared_data.local_src[local_cell * local_dim + index] =
  //             src[global_dof_indices];
  //
  //           shared_data.local_dst[local_cell * local_dim + index] =
  //             dst[global_dof_indices];
  //         }
  //
  //       evaluate_smooth_cf<dim,
  //                          fe_degree,
  //                          Number,
  //                          lapalace,
  //                          SmootherVariant::ConflictFree>(local_cell,
  //                                                         &shared_data,
  //                                                         &gpu_data);
  //
  //       if (dim == 2)
  //         {
  //           unsigned int linear_tid = local_tid_x + threadIdx.y * n_dofs_1d;
  //
  //           if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
  //             {
  //               int row = linear_tid / (n_dofs_1d - 2) + 1;
  //               int col = linear_tid % (n_dofs_1d - 2) + 1;
  //
  //               const unsigned int index = 2 * local_cell * local_dim +
  //                                          (row - 1) * (n_dofs_1d - 2) + col
  //                                          - 1;
  //
  //               const unsigned int global_index =
  //                 Util::compute_indices<dim, fe_degree>(
  //                   &gpu_data.first_dof[cell * regular_vpatch_size],
  //                   local_cell,
  //                   col,
  //                   row,
  //                   0);
  //
  //               const unsigned int global_dof_indices =
  //                 gpu_data.global_to_local(global_index);
  //
  //               solution[global_dof_indices] =
  //                 shared_data.tmp[index] * gpu_data.relaxation;
  //             }
  //         }
  //       else if (dim == 3)
  //         {
  //           for (unsigned int z = 0; z < n_dofs_1d - 2; ++z)
  //             {
  //               unsigned int linear_tid = local_tid_x + threadIdx.y *
  //               n_dofs_1d;
  //
  //               if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
  //                 {
  //                   unsigned int row = linear_tid / (n_dofs_1d - 2);
  //                   unsigned int col = linear_tid % (n_dofs_1d - 2);
  //
  //                   unsigned int index_g =
  //                     z * (n_dofs_1d - 2) * (n_dofs_1d - 2) +
  //                     row * (n_dofs_1d - 2) + col;
  //                   unsigned int index =
  //                     z * n_dofs_1d * n_dofs_1d + row * (n_dofs_1d - 2) +
  //                     col;
  //
  //                   types::global_dof_index global_dof_indices;
  //
  //                   if constexpr (is_ghost)
  //                     global_dof_indices =
  //                       gpu_data.patch_dofs[cell * local_dim +
  //                                           gpu_data.l_to_h[index_g]];
  //                   else
  //                     {
  //                       const types::global_dof_index global_index =
  //                         Util::compute_indices<dim, fe_degree>(
  //                           &gpu_data.first_dof[cell * regular_vpatch_size],
  //                           local_cell,
  //                           col + 1,
  //                           row + 1,
  //                           z + 1);
  //                       global_dof_indices =
  //                         gpu_data.global_to_local(global_index);
  //                     }
  //
  //                   solution[global_dof_indices] =
  //                     shared_data
  //                       .tmp[(dim - 1) * local_cell * local_dim + index] *
  //                     gpu_data.relaxation;
  //                 }
  //             }
  //         }
  //     }
  // }
  //
  // template <int dim, int fe_degree, typename Number, LaplaceVariant lapalace>
  // __global__ void
  // loop_kernel_fused_exact(
  //   const Number                                                 *src,
  //   Number                                                       *dst,
  //   Number                                                       *solution,
  //   const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  // {
  //   constexpr unsigned int n_dofs_1d           = 2 * fe_degree + 2;
  //   constexpr unsigned int local_dim           = Util::pow(n_dofs_1d, dim);
  //   constexpr unsigned int regular_vpatch_size = Util::pow(2, dim);
  //   constexpr unsigned int n_dofs_z            = dim == 2 ? 1 : n_dofs_1d;
  //
  //   const unsigned int cell_per_block = gpu_data.cell_per_block;
  //   const unsigned int local_cell     = threadIdx.x / n_dofs_1d;
  //   const unsigned int cell       = local_cell + cell_per_block *
  //   blockIdx.x; const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;
  //
  //   SharedMemData<dim, Number, false>
  //   shared_data(get_shared_data_ptr<Number>(),
  //                                                 cell_per_block,
  //                                                 n_dofs_1d,
  //                                                 local_dim);
  //
  //   if (cell < gpu_data.n_cells)
  //     {
  //       shared_data.local_mass[threadIdx.y * n_dofs_1d + local_tid_x] =
  //         gpu_data.smooth_mass_1d[threadIdx.y * n_dofs_1d + local_tid_x];
  //       shared_data.local_derivative[threadIdx.y * n_dofs_1d + local_tid_x] =
  //         gpu_data.smooth_stiff_1d[threadIdx.y * n_dofs_1d + local_tid_x];
  //
  //       for (unsigned int z = 0; z < n_dofs_z; ++z)
  //         {
  //           const unsigned int index = local_cell * local_dim +
  //                                      z * n_dofs_1d * n_dofs_1d +
  //                                      threadIdx.y * n_dofs_1d + local_tid_x;
  //
  //           const unsigned int global_index =
  //             Util::compute_indices<dim, fe_degree>(
  //               &gpu_data.first_dof[cell * regular_vpatch_size],
  //               local_cell,
  //               local_tid_x,
  //               threadIdx.y,
  //               z);
  //
  //           const unsigned int global_dof_indices =
  //             gpu_data.global_to_local(global_index);
  //
  //           shared_data.local_src[index] = src[global_dof_indices];
  //
  //           shared_data.local_dst[index] = dst[global_dof_indices];
  //         }
  //
  //       evaluate_smooth_exact<dim,
  //                             fe_degree,
  //                             Number,
  //                             lapalace,
  //                             SmootherVariant::ExactRes>(local_cell,
  //                                                        &shared_data,
  //                                                        &gpu_data);
  //
  //       const unsigned int linear_tid = local_tid_x + threadIdx.y *
  //       n_dofs_1d; if (dim == 2)
  //         {
  //           if (linear_tid < (n_dofs_1d - 4) * (n_dofs_1d - 4))
  //             {
  //               int row = linear_tid / (n_dofs_1d - 4) + 2;
  //               int col = linear_tid % (n_dofs_1d - 4) + 2;
  //
  //               const unsigned int index =
  //                 local_cell * local_dim + row * n_dofs_1d + col;
  //
  //               const unsigned int global_index =
  //                 Util::compute_indices<dim, fe_degree>(
  //                   &gpu_data.first_dof[cell * regular_vpatch_size],
  //                   local_cell,
  //                   col,
  //                   row,
  //                   0);
  //
  //               const unsigned int global_dof_indices =
  //                 gpu_data.global_to_local(global_index);
  //
  //               solution[global_dof_indices] =
  //                 shared_data.local_dst[index] * gpu_data.relaxation;
  //             }
  //         }
  //       else if (dim == 3)
  //         {
  //           if (linear_tid < (n_dofs_1d - 4) * (n_dofs_1d - 4))
  //             for (unsigned int z = 2; z < n_dofs_1d - 2; ++z)
  //               {
  //                 int row = linear_tid / (n_dofs_1d - 4) + 2;
  //                 int col = linear_tid % (n_dofs_1d - 4) + 2;
  //
  //                 unsigned int index = local_cell * local_dim +
  //                                      z * n_dofs_1d * n_dofs_1d +
  //                                      row * n_dofs_1d + col;
  //
  //                 const unsigned int global_index =
  //                   Util::compute_indices<dim, fe_degree>(
  //                     &gpu_data.first_dof[cell * regular_vpatch_size],
  //                     local_cell,
  //                     col,
  //                     row,
  //                     z);
  //
  //                 const unsigned int global_dof_indices =
  //                   gpu_data.global_to_local(global_index);
  //
  //                 solution[global_dof_indices] =
  //                   shared_data.local_dst[index] * gpu_data.relaxation;
  //               }
  //         }
  //     }
  // }

} // namespace PSMF

#endif // CELL_LOOP_KERNEL_CUH
