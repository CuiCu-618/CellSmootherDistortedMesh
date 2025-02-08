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

  template <int dim, int fe_degree, typename Number>
  __global__ void
  laplace_kernel_matrix(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {}


  template <int dim, int fe_degree, typename Number>
  __global__ void
  laplace_kernel_basic(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr int n_dofs_1d = 2 * fe_degree + 3;
    constexpr int n_dofs_2d = n_dofs_1d * n_dofs_1d;

    constexpr int n_patch_dofs_rt =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 3);
    constexpr int n_patch_dofs_dg = Util::pow(2 * fe_degree + 2, dim);
    constexpr int n_patch_dofs    = n_patch_dofs_rt + n_patch_dofs_dg;
    constexpr int block_size      = n_dofs_2d * dim;

    constexpr int n_first_dof_rt = n_first_dofs_rt<dim>();
    constexpr int n_first_dof_dg = 1 << dim;

    const int patch_per_block = gpu_data.patch_per_block;
    const int local_patch     = threadIdx.y / (n_dofs_1d * dim);
    const int patch           = local_patch + patch_per_block * blockIdx.x;

    const int tid_y     = threadIdx.y % (n_dofs_1d * dim);
    const int tid_x     = threadIdx.x;
    const int tid       = tid_y * n_dofs_1d + tid_x;
    const int tid_c     = (threadIdx.y % n_dofs_1d) * n_dofs_1d + tid_x;
    const int component = (threadIdx.y / n_dofs_1d) % dim;

    SharedDataOp<dim, Number, LaplaceVariant::Basic> shared_data(
      get_shared_data_ptr<Number>(), patch_per_block, n_dofs_1d, n_patch_dofs);

    if (patch < gpu_data.n_patches)
      {
        // L M
        for (int d = 0; d < dim; ++d)
          if ((d == 0 && tid_c < n_dofs_2d) ||
              (d != 0 && tid_c < (n_dofs_1d - 1) * (n_dofs_1d - 1)))
            {
              auto dd = component == 0 ? d :
                        component == 1 ? (d == 0 ? 1 :
                                          d == 1 ? 0 :
                                                   2) :
                                         (d == 0 ? 2 :
                                          d == 1 ? 0 :
                                                   1);

              const int shift = local_patch * n_dofs_2d * dim * dim;
              shared_data
                .local_mass[shift + (component * dim + d) * n_dofs_2d + tid_c] =
                gpu_data.rt_mass_1d[d][gpu_data.patch_type[patch * dim + dd] *
                                         n_dofs_2d +
                                       tid_c];
              shared_data
                .local_laplace[shift + (component * dim + d) * n_dofs_2d +
                               tid_c] =
                gpu_data
                  .rt_laplace_1d[d][gpu_data.patch_type[patch * dim + dd] *
                                      n_dofs_2d +
                                    tid_c];
            }

        {
          // D
          const int shift1 = local_patch * n_dofs_2d * dim;
          if (tid_c < n_dofs_1d * (n_dofs_1d - 1))
            shared_data.local_mix_der[shift1 + component * n_dofs_2d + tid_c] =
              -gpu_data
                 .mix_der_1d[0][gpu_data.patch_type[patch * dim + component] *
                                  n_dofs_2d +
                                tid_c];
          // M
          const int shift2 = local_patch * n_dofs_2d * dim * (dim - 1);
          auto      dd1    = component == 0 ? 1 : 0;
          if (tid_c < (n_dofs_1d - 1) * (n_dofs_1d - 1))
            shared_data
              .local_mix_mass[shift2 + component * n_dofs_2d * (dim - 1) +
                              tid_c] =
              gpu_data.mix_mass_1d[1][gpu_data.patch_type[patch * dim + dd1] *
                                        n_dofs_2d +
                                      tid_c];
          if (dim == 3)
            {
              auto dd2 = component == 2 ? 1 : 2;
              if (tid_c < (n_dofs_1d - 1) * (n_dofs_1d - 1))
                shared_data
                  .local_mix_mass[shift2 + component * n_dofs_2d * (dim - 1) +
                                  n_dofs_2d + tid_c] =
                  gpu_data
                    .mix_mass_1d[2][gpu_data.patch_type[patch * dim + dd2] *
                                      n_dofs_2d +
                                    tid_c];
            }
        }


        for (unsigned int i = 0; i < n_patch_dofs_rt / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_rt)
            {
              // unsigned int global_dof_index =
              //   gpu_data
              //     .patch_dof_laplace[patch * n_patch_dofs +
              //                        gpu_data.htol_rt[tid + i * block_size]];

              unsigned int global_dof_index =
                gpu_data
                  .first_dof_rt[patch * n_first_dof_rt +
                                gpu_data.base_dof_rt
                                  [gpu_data.htol_rt[tid + i * block_size]]] +
                gpu_data.dof_offset_rt[gpu_data.htol_rt[tid + i * block_size]];

              shared_data
                .local_src[local_patch * n_patch_dofs + tid + i * block_size] =
                src[global_dof_index];
            }
        for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_dg)
            {
              // unsigned int global_dof_index_n =
              //   gpu_data
              //     .patch_dof_laplace[patch * n_patch_dofs + n_patch_dofs_rt +
              //                        gpu_data.htol_dgn[tid + i *
              //                        block_size]];

              unsigned int global_dof_index_n =
                gpu_data
                  .first_dof_dg[patch * n_first_dof_dg +
                                gpu_data.base_dof_dg
                                  [gpu_data.htol_dgn[tid + i * block_size]]] +
                gpu_data.dof_offset_dg[gpu_data.htol_dgn[tid + i * block_size]];

              shared_data.local_src[local_patch * n_patch_dofs +
                                    n_patch_dofs_rt + tid + i * block_size] =
                src[global_dof_index_n];
            }


        evaluate_laplace<dim, fe_degree, Number, LaplaceVariant::Basic>(
          local_patch, &shared_data, &gpu_data);
        __syncthreads();

        for (unsigned int i = 0; i < n_patch_dofs_rt / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_rt)
            {
              // unsigned int global_dof_index =
              //   gpu_data
              //     .patch_dof_laplace[patch * n_patch_dofs +
              //                        gpu_data.htol_rt[tid + i * block_size]];

              unsigned int global_dof_index =
                gpu_data
                  .first_dof_rt[patch * n_first_dof_rt +
                                gpu_data.base_dof_rt
                                  [gpu_data.htol_rt[tid + i * block_size]]] +
                gpu_data.dof_offset_rt[gpu_data.htol_rt[tid + i * block_size]];

              atomicAdd(&dst[global_dof_index],
                        shared_data.local_dst[local_patch * n_patch_dofs + tid +
                                              i * block_size]);
            }
        __syncthreads();
        for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_dg)
            {
              // unsigned int global_dof_index =
              //   gpu_data
              //     .patch_dof_laplace[patch * n_patch_dofs + n_patch_dofs_rt +
              //                        tid + i * block_size];

              unsigned int global_dof_index =
                gpu_data
                  .first_dof_dg[patch * n_first_dof_dg +
                                gpu_data.base_dof_dg[tid + i * block_size]] +
                gpu_data.dof_offset_dg[tid + i * block_size];

              atomicAdd(
                &dst[global_dof_index],
                shared_data.local_dst[local_patch * n_patch_dofs +
                                      n_patch_dofs_rt + tid + i * block_size]);
            }
      }
  }


  template <int dim, int fe_degree, typename Number>
  __global__ void __launch_bounds__(64, 1) laplace_kernel_tensorcore(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr int n_dofs_1d = 2 * fe_degree + 3;
    constexpr int n_dofs_2d = n_dofs_1d * n_dofs_1d;

    constexpr int n_patch_dofs_rt =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 3);
    constexpr int n_patch_dofs_dg = Util::pow(2 * fe_degree + 2, dim);
    constexpr int n_patch_dofs    = n_patch_dofs_rt + n_patch_dofs_dg;
    constexpr int block_size      = n_dofs_2d;

    constexpr int n_first_dof_rt = n_first_dofs_rt<dim>();
    constexpr int n_first_dof_dg = 1 << dim;

    const int patch_per_block = 1;
    const int local_patch     = 0;
    const int patch           = blockIdx.x;

    const int tid_y = threadIdx.y;
    const int tid_x = threadIdx.x;
    const int tid   = tid_y * (n_dofs_1d + 1) + tid_x;
    const int tid_c = (threadIdx.y % (n_dofs_1d + 1)) * (n_dofs_1d + 1) + tid_x;
    // const int component = (threadIdx.y / (n_dofs_1d + 1)) % dim;

    SharedDataOp<dim, Number, LaplaceVariant::Basic> shared_data(
      get_shared_data_ptr<Number>(),
      patch_per_block,
      n_dofs_1d + 1,
      n_patch_dofs);

    if (patch < gpu_data.n_patches)
      {
        // L M
        for (int component = 0; component < dim; ++component)
          for (int d = 0; d < dim; ++d)
            if ((d == 0 && tid_c < n_dofs_2d) ||
                (d != 0 && tid_c < (n_dofs_1d - 1) * (n_dofs_1d - 1)))
              {
                auto dd = component == 0 ? d :
                          component == 1 ? (d == 0 ? 1 :
                                            d == 1 ? 0 :
                                                     2) :
                                           (d == 0 ? 2 :
                                            d == 1 ? 0 :
                                                     1);

                const int shift = local_patch * n_dofs_2d * dim * dim;
                shared_data
                  .local_mass[shift + (component * dim + d) * n_dofs_2d +
                              tid_c] =
                  gpu_data.rt_mass_1d[d][gpu_data.patch_type[patch * dim + dd] *
                                           n_dofs_2d +
                                         tid_c];
                shared_data
                  .local_laplace[shift + (component * dim + d) * n_dofs_2d +
                                 tid_c] =
                  gpu_data
                    .rt_laplace_1d[d][gpu_data.patch_type[patch * dim + dd] *
                                        n_dofs_2d +
                                      tid_c];
              }

        for (int component = 0; component < dim; ++component)
          {
            // D
            const int shift1 = local_patch * n_dofs_2d * dim;
            if (tid_c < n_dofs_1d * (n_dofs_1d - 1))
              shared_data
                .local_mix_der[shift1 + component * n_dofs_2d + tid_c] =
                -gpu_data
                   .mix_der_1d[0][gpu_data.patch_type[patch * dim + component] *
                                    n_dofs_2d +
                                  tid_c];
            // M
            const int shift2 = local_patch * n_dofs_2d * dim * (dim - 1);
            auto      dd1    = component == 0 ? 1 : 0;
            if (tid_c < (n_dofs_1d - 1) * (n_dofs_1d - 1))
              shared_data
                .local_mix_mass[shift2 + component * n_dofs_2d * (dim - 1) +
                                tid_c] =
                gpu_data.mix_mass_1d[1][gpu_data.patch_type[patch * dim + dd1] *
                                          n_dofs_2d +
                                        tid_c];
            if (dim == 3)
              {
                auto dd2 = component == 2 ? 1 : 2;
                if (tid_c < (n_dofs_1d - 1) * (n_dofs_1d - 1))
                  shared_data
                    .local_mix_mass[shift2 + component * n_dofs_2d * (dim - 1) +
                                    n_dofs_2d + tid_c] =
                    gpu_data
                      .mix_mass_1d[2][gpu_data.patch_type[patch * dim + dd2] *
                                        n_dofs_2d +
                                      tid_c];
              }
          }


        for (unsigned int i = 0; i < n_patch_dofs_rt / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_rt)
            {
              unsigned int global_dof_index =
                gpu_data
                  .first_dof_rt[patch * n_first_dof_rt +
                                gpu_data.base_dof_rt
                                  [gpu_data.htol_rt[tid + i * block_size]]] +
                gpu_data.dof_offset_rt[gpu_data.htol_rt[tid + i * block_size]];

              shared_data
                .local_src[local_patch * n_patch_dofs + tid + i * block_size] =
                src[global_dof_index];
            }
        for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_dg)
            {
              unsigned int global_dof_index_n =
                gpu_data
                  .first_dof_dg[patch * n_first_dof_dg +
                                gpu_data.base_dof_dg
                                  [gpu_data.htol_dgn[tid + i * block_size]]] +
                gpu_data.dof_offset_dg[gpu_data.htol_dgn[tid + i * block_size]];

              shared_data.local_src[local_patch * n_patch_dofs +
                                    n_patch_dofs_rt + tid + i * block_size] =
                src[global_dof_index_n];
            }

        evaluate_laplace_tensorcore<dim,
                                    fe_degree,
                                    Number,
                                    LaplaceVariant::Basic>(local_patch,
                                                           &shared_data,
                                                           &gpu_data);
        __syncthreads();

        for (unsigned int i = 0; i < n_patch_dofs_rt / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_rt)
            {
              unsigned int global_dof_index =
                gpu_data
                  .first_dof_rt[patch * n_first_dof_rt +
                                gpu_data.base_dof_rt
                                  [gpu_data.htol_rt[tid + i * block_size]]] +
                gpu_data.dof_offset_rt[gpu_data.htol_rt[tid + i * block_size]];

              atomicAdd(&dst[global_dof_index],
                        shared_data.local_dst[local_patch * n_patch_dofs + tid +
                                              i * block_size]);
            }
        __syncthreads();
        for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_dg)
            {
              unsigned int global_dof_index =
                gpu_data
                  .first_dof_dg[patch * n_first_dof_dg +
                                gpu_data.base_dof_dg[tid + i * block_size]] +
                gpu_data.dof_offset_dg[tid + i * block_size];

              atomicAdd(
                &dst[global_dof_index],
                shared_data.local_dst[local_patch * n_patch_dofs +
                                      n_patch_dofs_rt + tid + i * block_size]);
            }
      }
  }


  template <int dim, int fe_degree, typename Number>
  __global__ void __launch_bounds__(192, 1) laplace_kernel_tensorcore_padding(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr int n_dofs_1d = 2 * fe_degree + 3;
    constexpr int n_dofs_2d = n_dofs_1d * n_dofs_1d;

    constexpr int n_patch_dofs_rt =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 3);
    constexpr int n_patch_dofs_dg = Util::pow(2 * fe_degree + 2, dim);
    constexpr int n_patch_dofs    = n_patch_dofs_rt + n_patch_dofs_dg;
    constexpr int block_size      = n_dofs_2d * dim;

    constexpr int n_first_dof_rt = n_first_dofs_rt<dim>();
    constexpr int n_first_dof_dg = 1 << dim;

    const int patch_per_block = gpu_data.patch_per_block;
    const int local_patch     = threadIdx.y / ((n_dofs_1d + 1) * dim);
    const int patch           = local_patch + patch_per_block * blockIdx.x;

    const int tid_y = threadIdx.y % ((n_dofs_1d + 1) * dim);
    const int tid_x = threadIdx.x;
    const int tid   = tid_y * (n_dofs_1d + 1) + tid_x;
    const int tid_c = (threadIdx.y % (n_dofs_1d + 1)) * (n_dofs_1d + 1) + tid_x;
    const int component = (threadIdx.y / (n_dofs_1d + 1)) % dim;

    SharedDataOp<dim, Number, LaplaceVariant::Basic> shared_data(
      get_shared_data_ptr<Number>(),
      patch_per_block,
      n_dofs_1d + 1,
      (dim + 1) * Util::pow(2 * fe_degree + 4, dim));

    if (patch < gpu_data.n_patches)
      {
        // L M
        for (int d = 0; d < dim; ++d)
          if ((d == 0 && tid_c < n_dofs_2d) ||
              (d != 0 && tid_c < (n_dofs_1d - 1) * (n_dofs_1d - 1)))
            {
              auto dd = component == 0 ? d :
                        component == 1 ? (d == 0 ? 1 :
                                          d == 1 ? 0 :
                                                   2) :
                                         (d == 0 ? 2 :
                                          d == 1 ? 0 :
                                                   1);

              const int shift = local_patch * n_dofs_2d * dim * dim;
              shared_data
                .local_mass[shift + (component * dim + d) * n_dofs_2d + tid_c] =
                gpu_data.rt_mass_1d[d][gpu_data.patch_type[patch * dim + dd] *
                                         n_dofs_2d +
                                       tid_c];
              shared_data
                .local_laplace[shift + (component * dim + d) * n_dofs_2d +
                               tid_c] =
                gpu_data
                  .rt_laplace_1d[d][gpu_data.patch_type[patch * dim + dd] *
                                      n_dofs_2d +
                                    tid_c];
            }

        {
          // D
          const int shift1 = local_patch * n_dofs_2d * dim;
          if (tid_c < n_dofs_1d * (n_dofs_1d - 1))
            shared_data.local_mix_der[shift1 + component * n_dofs_2d + tid_c] =
              -gpu_data
                 .mix_der_1d[0][gpu_data.patch_type[patch * dim + component] *
                                  n_dofs_2d +
                                tid_c];
          // M
          const int shift2 = local_patch * n_dofs_2d * dim * (dim - 1);
          auto      dd1    = component == 0 ? 1 : 0;
          if (tid_c < (n_dofs_1d - 1) * (n_dofs_1d - 1))
            shared_data
              .local_mix_mass[shift2 + component * n_dofs_2d * (dim - 1) +
                              tid_c] =
              gpu_data.mix_mass_1d[1][gpu_data.patch_type[patch * dim + dd1] *
                                        n_dofs_2d +
                                      tid_c];
          if (dim == 3)
            {
              auto dd2 = component == 2 ? 1 : 2;
              if (tid_c < (n_dofs_1d - 1) * (n_dofs_1d - 1))
                shared_data
                  .local_mix_mass[shift2 + component * n_dofs_2d * (dim - 1) +
                                  n_dofs_2d + tid_c] =
                  gpu_data
                    .mix_mass_1d[2][gpu_data.patch_type[patch * dim + dd2] *
                                      n_dofs_2d +
                                    tid_c];
            }
        }


        for (unsigned int i = 0; i < n_patch_dofs_rt / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_rt)
            {
              unsigned int global_dof_index =
                gpu_data
                  .first_dof_rt[patch * n_first_dof_rt +
                                gpu_data.base_dof_rt
                                  [gpu_data.htol_rt[tid + i * block_size]]] +
                gpu_data.dof_offset_rt[gpu_data.htol_rt[tid + i * block_size]];

              shared_data
                .local_src[local_patch * n_patch_dofs + tid + i * block_size] =
                src[global_dof_index];
            }
        for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_dg)
            {
              unsigned int global_dof_index_n =
                gpu_data
                  .first_dof_dg[patch * n_first_dof_dg +
                                gpu_data.base_dof_dg
                                  [gpu_data.htol_dgn[tid + i * block_size]]] +
                gpu_data.dof_offset_dg[gpu_data.htol_dgn[tid + i * block_size]];

              shared_data.local_src[local_patch * n_patch_dofs +
                                    n_patch_dofs_rt + tid + i * block_size] =
                src[global_dof_index_n];
            }


        evaluate_laplace_tensorcore_padding<dim,
                                            fe_degree,
                                            Number,
                                            LaplaceVariant::Basic>(local_patch,
                                                                   &shared_data,
                                                                   &gpu_data);
        __syncthreads();

        for (unsigned int i = 0; i < n_patch_dofs_rt / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_rt)
            {
              unsigned int global_dof_index =
                gpu_data
                  .first_dof_rt[patch * n_first_dof_rt +
                                gpu_data.base_dof_rt
                                  [gpu_data.htol_rt[tid + i * block_size]]] +
                gpu_data.dof_offset_rt[gpu_data.htol_rt[tid + i * block_size]];

              atomicAdd(&dst[global_dof_index],
                        shared_data.local_dst[local_patch * n_patch_dofs + tid +
                                              i * block_size]);
            }
        __syncthreads();
        for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_dg)
            {
              unsigned int global_dof_index =
                gpu_data
                  .first_dof_dg[patch * n_first_dof_dg +
                                gpu_data.base_dof_dg[tid + i * block_size]] +
                gpu_data.dof_offset_dg[tid + i * block_size];

              atomicAdd(
                &dst[global_dof_index],
                shared_data.local_dst[local_patch * n_patch_dofs +
                                      n_patch_dofs_rt + tid + i * block_size]);
            }
      }
  }



  template <int dim, int fe_degree, typename Number>
  __global__ void
  laplace_kernel_basicpadding(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr int n_dofs_1d   = 2 * fe_degree + 3;
    constexpr int n_dofs_2d   = n_dofs_1d * n_dofs_1d;
    constexpr int n_dofs_1d_z = dim == 2 ? 1 : n_dofs_1d - 1;

    constexpr int n_dofs_component =
      Util::pow(n_dofs_1d, dim - 1) * (2 * fe_degree + 2);
    constexpr int n_patch_dofs_rt =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 3);
    constexpr int n_patch_dofs_dg = Util::pow(2 * fe_degree + 2, dim);
    constexpr int n_patch_dofs    = n_dofs_component * (dim + 1);

    constexpr int n_first_dof_rt = n_first_dofs_rt<dim>();
    constexpr int n_first_dof_dg = 1 << dim;

    const int patch_per_block = gpu_data.patch_per_block;
    const int local_patch     = threadIdx.y / (n_dofs_1d * dim);
    const int patch           = local_patch + patch_per_block * blockIdx.x;

    const int    tid_y     = threadIdx.y % n_dofs_1d;
    const int    tid_x     = threadIdx.x;
    const int    tid       = tid_y * n_dofs_1d + tid_x;
    unsigned int component = (threadIdx.y / n_dofs_1d) % dim;

    SharedDataOp<dim, Number, LaplaceVariant::BasicPadding> shared_data(
      get_shared_data_ptr<Number>(), patch_per_block, n_dofs_1d, n_patch_dofs);

    if (patch < gpu_data.n_patches)
      {
        // L M
        if (component == 0)
          for (int dir = 0; dir < dim; ++dir)
            for (int d = 0; d < dim; ++d)
              {
                auto dd = dir == 0 ? d :
                          dir == 1 ? (d == 0 ? 1 :
                                      d == 1 ? 0 :
                                               2) :
                                     (d == 0 ? 2 :
                                      d == 1 ? 0 :
                                               1);

                const int shift = local_patch * n_dofs_2d * dim * dim;
                shared_data
                  .local_mass[shift + (dir * dim + d) * n_dofs_2d + tid] =
                  gpu_data
                    .rt_mass_1d_p[d][gpu_data.patch_type[patch * dim + dd] *
                                       n_dofs_2d +
                                     tid];
                shared_data
                  .local_laplace[shift + (dir * dim + d) * n_dofs_2d + tid] =
                  gpu_data
                    .rt_laplace_1d_p[d][gpu_data.patch_type[patch * dim + dd] *
                                          n_dofs_2d +
                                        tid];
              }

        if (component == 0)
          for (int dir = 0; dir < dim; ++dir)
            {
              // D
              const int shift1 = local_patch * n_dofs_2d * dim;
              shared_data.local_mix_der[shift1 + dir * n_dofs_2d + tid] =
                -gpu_data
                   .mix_der_1d_p[0][gpu_data.patch_type[patch * dim + dir] *
                                      n_dofs_2d +
                                    tid];
              // M
              const int shift2 = local_patch * n_dofs_2d * dim * (dim - 1);
              auto      dd1    = dir == 0 ? 1 : 0;
              shared_data
                .local_mix_mass[shift2 + dir * n_dofs_2d * (dim - 1) + tid] =
                gpu_data.mix_mass_1d_p
                  [1][gpu_data.patch_type[patch * dim + dd1] * n_dofs_2d + tid];
              if (dim == 3)
                {
                  auto dd2 = dir == 2 ? 1 : 2;
                  shared_data
                    .local_mix_mass[shift2 + dir * n_dofs_2d * (dim - 1) +
                                    n_dofs_2d + tid] =
                    gpu_data
                      .mix_mass_1d_p[2][gpu_data.patch_type[patch * dim + dd2] *
                                          n_dofs_2d +
                                        tid];
                }
            }

        // // debug
        // __syncthreads();
        // if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
        //   {
        //     auto shift = n_dofs_2d;
        //     printf("mass\n");
        //     for (unsigned int i = 0; i < n_dofs_1d; ++i)
        //       {
        //         for (unsigned int j = 0; j < n_dofs_1d; ++j)
        //           printf("%f ",
        //                  shared_data.local_mass[shift + i * n_dofs_2d + j]);
        //         printf("\n");
        //       }
        //     printf("der\n");
        //     for (unsigned int i = 0; i < n_dofs_1d; ++i)
        //       {
        //         for (unsigned int j = 0; j < n_dofs_1d; ++j)
        //           printf("%f ",
        //                  shared_data.local_laplace[shift + i * n_dofs_2d +
        //                  j]);
        //         printf("\n");
        //       }
        //     printf("mixm\n");
        //     for (unsigned int i = 0; i < n_dofs_1d; ++i)
        //       {
        //         for (unsigned int j = 0; j < n_dofs_1d; ++j)
        //           printf("%f ",
        //                  shared_data.local_mix_mass[shift + i * n_dofs_2d +
        //                  j]);
        //         printf("\n");
        //       }
        //     printf("mixd\n");
        //     for (unsigned int i = 0; i < n_dofs_1d; ++i)
        //       {
        //         for (unsigned int j = 0; j < n_dofs_1d; ++j)
        //           printf("%f ",
        //                  shared_data.local_mix_der[shift + i * n_dofs_2d +
        //                  j]);
        //         printf("\n");
        //       }
        //   }

        // RT
        for (unsigned int z = 0; z < n_dofs_1d_z; ++z)
          {
            unsigned int index = component * n_patch_dofs_rt / dim +
                                 z * n_dofs_1d * (n_dofs_1d - 1) +
                                 (threadIdx.y % n_dofs_1d) * n_dofs_1d +
                                 threadIdx.x;

            if (tid_y < n_dofs_1d - 1)
              {
                unsigned int global_dof_index =
                  gpu_data.first_dof_rt
                    [patch * n_first_dof_rt +
                     gpu_data.base_dof_rt[gpu_data.htol_rt[index]]] +
                  gpu_data.dof_offset_rt[gpu_data.htol_rt[index]];

                shared_data.local_src[local_patch * n_patch_dofs +
                                      component * n_dofs_component +
                                      z * n_dofs_1d * n_dofs_1d + tid] =
                  src[global_dof_index];
              }
            else
              shared_data.local_src[local_patch * n_patch_dofs +
                                    component * n_dofs_component +
                                    z * n_dofs_1d * n_dofs_1d + tid] = 0;
          }

        // DG
        if (component == 0)
          for (unsigned int z = 0; z < n_dofs_1d_z; ++z)
            {
              unsigned int index = z * (n_dofs_1d - 1) * (n_dofs_1d - 1) +
                                   (threadIdx.y % n_dofs_1d) * (n_dofs_1d - 1) +
                                   (threadIdx.x % n_dofs_1d);

              if (tid_x < n_dofs_1d - 1 && tid_y < n_dofs_1d - 1)
                {
                  unsigned int global_dof_index_n =
                    gpu_data.first_dof_dg
                      [patch * n_first_dof_dg +
                       gpu_data.base_dof_dg[gpu_data.htol_dgn[index]]] +
                    gpu_data.dof_offset_dg[gpu_data.htol_dgn[index]];

                  shared_data.local_src[local_patch * n_patch_dofs +
                                        dim * n_dofs_component +
                                        z * n_dofs_1d * n_dofs_1d + tid] =
                    src[global_dof_index_n];
                }
              else
                shared_data.local_src[local_patch * n_patch_dofs +
                                      dim * n_dofs_component +
                                      z * n_dofs_1d * n_dofs_1d + tid] = 0;
            }

        // // debug
        // __syncthreads();
        // if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
        //   {
        //     printf("rt\n");
        //     for (unsigned int d = 0; d < dim; ++d)
        //       for (unsigned int i = 0; i < n_dofs_1d; ++i)
        //         {
        //           for (unsigned int j = 0; j < n_dofs_1d; ++j)
        //             printf("%f ",
        //                    shared_data.local_src[d * n_dofs_component +
        //                                          i * n_dofs_1d + j]);
        //           printf("\n");
        //         }
        //     printf("dg\n");
        //     for (unsigned int i = 0; i < n_dofs_1d; ++i)
        //       {
        //         for (unsigned int j = 0; j < n_dofs_1d; ++j)
        //           printf("%f ",
        //                  shared_data.local_src[dim * n_dofs_component +
        //                                        i * n_dofs_1d + j]);
        //         printf("\n");
        //       }
        //   }

        evaluate_laplace_padding<dim,
                                 fe_degree,
                                 Number,
                                 LaplaceVariant::BasicPadding>(local_patch,
                                                               &shared_data,
                                                               &gpu_data);
        __syncthreads();


        // // debug
        // __syncthreads();
        // if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
        //   {
        //     printf("after rt\n");
        //     for (unsigned int d = 0; d < dim; ++d)
        //       for (unsigned int i = 0; i < n_dofs_1d; ++i)
        //         {
        //           for (unsigned int j = 0; j < n_dofs_1d; ++j)
        //             printf("%f ",
        //                    shared_data.local_dst[d * n_dofs_component +
        //                                          i * n_dofs_1d + j]);
        //           printf("\n");
        //         }
        //     printf("after dg\n");
        //     for (unsigned int i = 0; i < n_dofs_1d; ++i)
        //       {
        //         for (unsigned int j = 0; j < n_dofs_1d; ++j)
        //           printf("%f ",
        //                  shared_data.local_dst[dim * n_dofs_component +
        //                                        i * n_dofs_1d + j]);
        //         printf("\n");
        //       }
        //   }

        // RT
        for (unsigned int z = 0; z < n_dofs_1d_z; ++z)
          {
            unsigned int index = component * n_patch_dofs_rt / dim +
                                 z * n_dofs_1d * (n_dofs_1d - 1) +
                                 (threadIdx.y % n_dofs_1d) * n_dofs_1d +
                                 threadIdx.x;

            if (tid_y < n_dofs_1d - 1)
              {
                unsigned int global_dof_index =
                  gpu_data.first_dof_rt
                    [patch * n_first_dof_rt +
                     gpu_data.base_dof_rt[gpu_data.htol_rt[index]]] +
                  gpu_data.dof_offset_rt[gpu_data.htol_rt[index]];

                atomicAdd(
                  &dst[global_dof_index],
                  shared_data.local_dst[local_patch * n_patch_dofs +
                                        component * n_dofs_component +
                                        z * n_dofs_1d * n_dofs_1d + tid]);
              }
          }

        // DG
        if (component == 0)
          for (unsigned int z = 0; z < n_dofs_1d_z; ++z)
            {
              unsigned int index = z * (n_dofs_1d - 1) * (n_dofs_1d - 1) +
                                   (threadIdx.y % n_dofs_1d) * (n_dofs_1d - 1) +
                                   (threadIdx.x % n_dofs_1d);

              if (tid_x < n_dofs_1d - 1 && tid_y < n_dofs_1d - 1)
                {
                  unsigned int global_dof_index_n =
                    gpu_data.first_dof_dg[patch * n_first_dof_dg +
                                          gpu_data.base_dof_dg[index]] +
                    gpu_data.dof_offset_dg[index];

                  atomicAdd(
                    &dst[global_dof_index_n],
                    shared_data.local_dst[local_patch * n_patch_dofs +
                                          dim * n_dofs_component +
                                          z * n_dofs_1d * n_dofs_1d + tid]);
                }
            }
      }
  }



  template <int dim, int fe_degree, typename Number>
  __global__ void
  laplace_kernel_cf(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr int n_dofs_1d   = 2 * fe_degree + 3;
    constexpr int n_dofs_2d   = n_dofs_1d * n_dofs_1d;
    constexpr int n_dofs_1d_z = dim == 2 ? 1 : n_dofs_1d - 1;

    constexpr int n_dofs_component =
      Util::pow(n_dofs_1d, dim - 1) * (2 * fe_degree + 2);
    constexpr int n_patch_dofs_rt =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 3);
    constexpr int n_patch_dofs_dg = Util::pow(2 * fe_degree + 2, dim);
    constexpr int n_patch_dofs    = n_dofs_component * (dim + 1);

    constexpr int n_first_dof_rt = n_first_dofs_rt<dim>();
    constexpr int n_first_dof_dg = 1 << dim;

    const int patch_per_block = gpu_data.patch_per_block;
    const int local_patch     = threadIdx.y / (n_dofs_1d * dim);
    const int patch           = local_patch + patch_per_block * blockIdx.x;

    const int    tid_y     = threadIdx.y % n_dofs_1d;
    const int    tid_x     = threadIdx.x;
    const int    tid       = tid_y * n_dofs_1d + tid_x;
    unsigned int component = (threadIdx.y / n_dofs_1d) % dim;

    SharedDataOp<dim, Number, LaplaceVariant::ConflictFree> shared_data(
      get_shared_data_ptr<Number>(), patch_per_block, n_dofs_1d, n_patch_dofs);

    if (patch < gpu_data.n_patches)
      {
        // L M
        for (int d = 0; d < dim; ++d)
          {
            auto dd = component == 0 ? d :
                      component == 1 ? (d == 0 ? 1 :
                                        d == 1 ? 0 :
                                                 2) :
                                       (d == 0 ? 2 :
                                        d == 1 ? 0 :
                                                 1);

            const int shift = local_patch * n_dofs_2d * dim * dim;
            shared_data
              .local_mass[shift + (component * dim + d) * n_dofs_2d + tid] =
              gpu_data.rt_mass_1d_p[d][gpu_data.patch_type[patch * dim + dd] *
                                         n_dofs_2d +
                                       tid];
            shared_data
              .local_laplace[shift + (component * dim + d) * n_dofs_2d + tid] =
              gpu_data.rt_laplace_1d_p
                [d][gpu_data.patch_type[patch * dim + dd] * n_dofs_2d + tid];
          }

        {
          // D
          const int shift1 = local_patch * n_dofs_2d * dim;
          shared_data.local_mix_der[shift1 + component * n_dofs_2d + tid] =
            -gpu_data
               .mix_der_1d_p[0][gpu_data.patch_type[patch * dim + component] *
                                  n_dofs_2d +
                                tid];
          // M
          const int shift2 = local_patch * n_dofs_2d * dim * (dim - 1);
          auto      dd1    = component == 0 ? 1 : 0;
          shared_data
            .local_mix_mass[shift2 + component * n_dofs_2d * (dim - 1) + tid] =
            gpu_data.mix_mass_1d_p[1][gpu_data.patch_type[patch * dim + dd1] *
                                        n_dofs_2d +
                                      tid];
          if (dim == 3)
            {
              auto dd2 = component == 2 ? 1 : 2;
              shared_data
                .local_mix_mass[shift2 + component * n_dofs_2d * (dim - 1) +
                                n_dofs_2d + tid] =
                gpu_data.mix_mass_1d_p
                  [2][gpu_data.patch_type[patch * dim + dd2] * n_dofs_2d + tid];
            }
        }
        // RT
        for (unsigned int z = 0; z < n_dofs_1d_z; ++z)
          {
            unsigned int index = component * n_patch_dofs_rt / dim +
                                 z * n_dofs_1d * (n_dofs_1d - 1) +
                                 (threadIdx.y % n_dofs_1d) * n_dofs_1d +
                                 threadIdx.x;

            shared_data
              .local_src[local_patch * n_patch_dofs + dim * n_dofs_component +
                         z * n_dofs_1d * n_dofs_1d + tid] = 0;

            if (tid_y < n_dofs_1d - 1)
              {
                unsigned int global_dof_index =
                  gpu_data.first_dof_rt
                    [patch * n_first_dof_rt +
                     gpu_data.base_dof_rt[gpu_data.htol_rt[index]]] +
                  gpu_data.dof_offset_rt[gpu_data.htol_rt[index]];

                shared_data.local_src[local_patch * n_patch_dofs +
                                      component * n_dofs_component +
                                      z * n_dofs_1d * n_dofs_1d + tid] =
                  src[global_dof_index];
              }
            else
              shared_data.local_src[local_patch * n_patch_dofs +
                                    component * n_dofs_component +
                                    z * n_dofs_1d * n_dofs_1d + tid] = 0;
          }

        __syncthreads();

        // DG
        if (component == 0)
          for (unsigned int z = 0; z < n_dofs_1d_z; ++z)
            {
              unsigned int index = z * (n_dofs_1d - 1) * (n_dofs_1d - 1) +
                                   (threadIdx.y % n_dofs_1d) * (n_dofs_1d - 1) +
                                   (threadIdx.x % n_dofs_1d);

              if (tid_x < n_dofs_1d - 1 && tid_y < n_dofs_1d - 1)
                {
                  unsigned int global_dof_index_n =
                    gpu_data.first_dof_dg
                      [patch * n_first_dof_dg +
                       gpu_data.base_dof_dg[gpu_data.htol_dgn[index]]] +
                    gpu_data.dof_offset_dg[gpu_data.htol_dgn[index]];

                  shared_data.local_src[local_patch * n_patch_dofs +
                                        dim * n_dofs_component +
                                        z * n_dofs_1d * n_dofs_1d + tid] =
                    src[global_dof_index_n];
                }
              else
                shared_data.local_src[local_patch * n_patch_dofs +
                                      dim * n_dofs_component +
                                      z * n_dofs_1d * n_dofs_1d + tid] = 0;
            }

        evaluate_laplace_padding<dim,
                                 fe_degree,
                                 Number,
                                 LaplaceVariant::ConflictFree>(local_patch,
                                                               &shared_data,
                                                               &gpu_data);
        __syncthreads();

        // RT
        for (unsigned int z = 0; z < n_dofs_1d_z; ++z)
          {
            unsigned int index = component * n_patch_dofs_rt / dim +
                                 z * n_dofs_1d * (n_dofs_1d - 1) +
                                 (threadIdx.y % n_dofs_1d) * n_dofs_1d +
                                 threadIdx.x;

            if (tid_y < n_dofs_1d - 1)
              {
                unsigned int global_dof_index =
                  gpu_data.first_dof_rt
                    [patch * n_first_dof_rt +
                     gpu_data.base_dof_rt[gpu_data.htol_rt[index]]] +
                  gpu_data.dof_offset_rt[gpu_data.htol_rt[index]];

                atomicAdd(
                  &dst[global_dof_index],
                  shared_data.local_dst[local_patch * n_patch_dofs +
                                        component * n_dofs_component +
                                        z * n_dofs_1d * n_dofs_1d + tid]);
              }
          }

        // DG
        if (component == 0)
          for (unsigned int z = 0; z < n_dofs_1d_z; ++z)
            {
              unsigned int index = z * (n_dofs_1d - 1) * (n_dofs_1d - 1) +
                                   (threadIdx.y % n_dofs_1d) * (n_dofs_1d - 1) +
                                   (threadIdx.x % n_dofs_1d);

              if (tid_x < n_dofs_1d - 1 && tid_y < n_dofs_1d - 1)
                {
                  unsigned int global_dof_index_n =
                    gpu_data.first_dof_dg[patch * n_first_dof_dg +
                                          gpu_data.base_dof_dg[index]] +
                    gpu_data.dof_offset_dg[index];

                  atomicAdd(
                    &dst[global_dof_index_n],
                    shared_data.local_dst[local_patch * n_patch_dofs +
                                          dim * n_dofs_component +
                                          z * n_dofs_1d * n_dofs_1d + tid]);
                }
            }
      }
  }



  template <int dim,
            int fe_degree,
            typename Number,
            LocalSolverVariant local_solver>
  __global__
    typename std::enable_if<local_solver == LocalSolverVariant::Direct>::type
    loop_kernel_global(
      const Number                                                 *src,
      Number                                                       *dst,
      const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_patch_dofs_all =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * (fe_degree + 2) - 1) +
      Util::pow(2 * fe_degree + 2, dim);

    constexpr unsigned int n_patch_dofs =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * (fe_degree + 2) - 3) +
      Util::pow(2 * fe_degree + 2, dim);

    const unsigned int tid   = threadIdx.x;
    const unsigned int patch = blockIdx.x;

    SharedDataSmoother<dim, Number, SmootherVariant::GLOBAL, local_solver>
      shared_data(get_shared_data_ptr<Number>(), 1, 0, n_patch_dofs);

    if (patch < gpu_data.n_patches)
      {
        for (unsigned int i = 0; i < n_patch_dofs / blockDim.x + 1; ++i)
          if (tid + i * blockDim.x < n_patch_dofs)
            {
              unsigned int global_dof_index =
                gpu_data
                  .patch_dof_smooth[patch * n_patch_dofs_all +
                                    gpu_data.h_interior[tid + i * blockDim.x]];

              shared_data.local_src[tid + i * blockDim.x] =
                src[global_dof_index];
              shared_data.local_dst[tid + i * blockDim.x] =
                dst[global_dof_index];
            }
        __syncthreads();

        constexpr unsigned int matrix_size = Util::pow(n_patch_dofs, 2);
        unsigned int           patch_type  = 0;
        for (unsigned int d = 0; d < dim; ++d)
          patch_type += gpu_data.patch_type[patch * dim + d] * Util::pow(3, d);

        for (unsigned int row = 0; row < n_patch_dofs; ++row)
          {
            for (unsigned int i = 0; i < n_patch_dofs / blockDim.x + 1; ++i)
              if (tid + i * blockDim.x < n_patch_dofs)
                {
                  auto val =
                    gpu_data
                      .eigenvalues[patch_type * matrix_size +
                                   row * n_patch_dofs + tid + i * blockDim.x] *
                    shared_data.local_src[tid + i * blockDim.x];

                  atomicAdd(&shared_data.local_dst[row], val);
                }
          }
        __syncthreads();

        for (unsigned int i = 0; i < n_patch_dofs / blockDim.x + 1; ++i)
          if (tid + i * blockDim.x < n_patch_dofs)
            {
              unsigned int global_dof_index =
                gpu_data
                  .patch_dof_smooth[patch * n_patch_dofs_all +
                                    gpu_data.h_interior[tid + i * blockDim.x]];

              dst[global_dof_index] =
                shared_data.local_dst[tid + i * blockDim.x] *
                gpu_data.relaxation;
            }
      }
  }

  template <int dim,
            int fe_degree,
            typename Number,
            LocalSolverVariant local_solver>
  __global__ typename std::enable_if<local_solver ==
                                     LocalSolverVariant::SchurDirect>::type
  loop_kernel_global(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {}

  template <int dim,
            int fe_degree,
            typename Number,
            LocalSolverVariant local_solver>
  __global__
    typename std::enable_if<local_solver ==
                            LocalSolverVariant::SchurTensorProduct>::type
    loop_kernel_global(
      const Number                                                 *src,
      Number                                                       *dst,
      const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr int n_dofs_1d = 2 * fe_degree + 3;
    constexpr int n_dofs_2d = n_dofs_1d * n_dofs_1d;

    constexpr int n_patch_dofs_rt =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 1);
    constexpr int n_patch_dofs_dg = Util::pow(2 * fe_degree + 2, dim);
    constexpr int n_patch_dofs    = n_patch_dofs_rt + n_patch_dofs_dg;
    constexpr int n_patch_dofs_rt_all =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 3);
    constexpr int n_patch_dofs_all =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * (fe_degree + 2) - 1) +
      Util::pow(2 * fe_degree + 2, dim);
    constexpr int block_size = n_dofs_2d * dim;

    constexpr int n_first_dof_rt = n_first_dofs_rt<dim>();
    constexpr int n_first_dof_dg = 1 << dim;

    const int patch_per_block = gpu_data.patch_per_block;
    const int local_patch     = threadIdx.y / (n_dofs_1d * dim);
    const int patch           = local_patch + patch_per_block * blockIdx.x;

    const int tid_y = threadIdx.y % (n_dofs_1d * dim);
    const int tid_x = threadIdx.x;
    const int tid   = tid_y * n_dofs_1d + tid_x;

    SharedDataSmoother<dim, Number, SmootherVariant::GLOBAL, local_solver>
      shared_data(get_shared_data_ptr<Number>(),
                  patch_per_block,
                  n_dofs_1d,
                  n_patch_dofs,
                  n_patch_dofs_dg);

    if (patch < gpu_data.n_patches)
      {
        unsigned int patch_type = 0;
        for (unsigned int d = 0; d < dim; ++d)
          patch_type += gpu_data.patch_type[patch * dim + d] * Util::pow(3, d);

        // eigs
        for (int dir = 0; dir < dim; ++dir)
          for (int d = 0; d < dim; ++d)
            if ((d == 0 && tid < (n_dofs_1d - 2) * (n_dofs_1d - 1)) ||
                (d != 0 && tid < (n_dofs_1d - 1) * (n_dofs_1d - 1)))
              {
                if ((d == 0 && tid < n_dofs_1d - 2) ||
                    (d != 0 && tid < n_dofs_1d - 1))
                  {
                    const int shift = local_patch * n_dofs_1d * dim * dim;
                    shared_data
                      .local_mass[shift + (dir * dim + d) * n_dofs_1d + tid] =
                      gpu_data.eigvals[dir][patch_type * n_dofs_1d * dim +
                                            d * n_dofs_1d + tid];
                  }
                const int shift2 = local_patch * n_dofs_2d * dim * dim;
                shared_data
                  .local_laplace[shift2 + (dir * dim + d) * n_dofs_2d + tid] =
                  gpu_data.eigvecs[dir][patch_type * n_dofs_2d * dim +
                                        d * n_dofs_2d + tid];
              }

        {
          // D
          const int shift1 = local_patch * n_dofs_2d * 1;
          if (tid < (n_dofs_1d - 2) * (n_dofs_1d - 1))
            shared_data.local_mix_der[shift1 + tid] =
              -gpu_data.smooth_mixmass_1d[0][tid];
        }

        for (unsigned int i = 0; i < n_patch_dofs_rt / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_rt)
            {
              // unsigned int global_dof_index =
              //   gpu_data.patch_dof_smooth
              //     [patch * n_patch_dofs_all +
              //      gpu_data.htol_rt_interior[tid + i * block_size]];

              unsigned int global_dof_index =
                gpu_data.first_dof_rt
                  [patch * n_first_dof_rt +
                   gpu_data.base_dof_rt
                     [gpu_data.htol_rt_interior[tid + i * block_size]]] +
                gpu_data.dof_offset_rt
                  [gpu_data.htol_rt_interior[tid + i * block_size]];

              shared_data
                .local_src[local_patch * n_patch_dofs + tid + i * block_size] =
                src[global_dof_index];
              shared_data
                .local_dst[local_patch * n_patch_dofs + tid + i * block_size] =
                0;
            }

        for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_dg)
            {
              // unsigned int global_dof_index =
              //   gpu_data
              //     .patch_dof_smooth[patch * n_patch_dofs_all +
              //                       n_patch_dofs_rt_all +
              //                       gpu_data.htol_dgn[tid + i * block_size]];

              unsigned int global_dof_index =
                gpu_data
                  .first_dof_dg[patch * n_first_dof_dg +
                                gpu_data.base_dof_dg
                                  [gpu_data.htol_dgn[tid + i * block_size]]] +
                gpu_data.dof_offset_dg[gpu_data.htol_dgn[tid + i * block_size]];

              shared_data.local_src[local_patch * n_patch_dofs +
                                    n_patch_dofs_rt + tid + i * block_size] =
                -src[global_dof_index];
              shared_data.local_dst[local_patch * n_patch_dofs +
                                    n_patch_dofs_rt + tid + i * block_size] = 0;
            }
        __syncthreads();

        /// B^T M^-1 B P = B^T M^-1 F - G
        // B^T M^-1 F - G

#ifdef TIMING
        __syncthreads();
        auto start_p = clock64();
#endif
        evaluate_smooth_p<dim, fe_degree, Number, decltype(shared_data)>(
          local_patch, &shared_data);
        __syncthreads();
#ifdef TIMING
        auto elapsed_p = clock64() - start_p;
        if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
          printf("Prepare P: %lld\n", elapsed_p);
#endif

#ifdef TIMING
        __syncthreads();
        auto start_cg = clock64();
#endif
        evaluate_smooth_cg<dim, fe_degree, Number, decltype(shared_data)>(
          local_patch, &shared_data);
        __syncthreads();
#ifdef TIMING
        auto elapsed_cg = clock64() - start_cg;
        if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
          printf("Solve P: %lld\n", elapsed_cg);

#endif

        for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_dg)
            {
              shared_data.local_dst[local_patch * n_patch_dofs +
                                    n_patch_dofs_rt + tid + i * block_size] =
                shared_data.local_src[local_patch * n_patch_dofs +
                                      ltoh_dgn[tid + i * block_size]];

              shared_data
                .local_src[local_patch * n_patch_dofs + n_patch_dofs_dg +
                           ltoh_dgt[tid + i * block_size]] =
                shared_data.local_src[local_patch * n_patch_dofs +
                                      ltoh_dgn[tid + i * block_size]];

              if constexpr (dim == 3)
                shared_data
                  .local_src[local_patch * n_patch_dofs + 2 * n_patch_dofs_dg +
                             ltoh_dgz[tid + i * block_size]] =
                  shared_data.local_src[local_patch * n_patch_dofs +
                                        ltoh_dgn[tid + i * block_size]];
            }

            /// M U = F - B P
#ifdef TIMING
        __syncthreads();
        auto start_u = clock64();
#endif
        evaluate_smooth_u<dim, fe_degree, Number, decltype(shared_data)>(
          local_patch, &shared_data);
        __syncthreads();
#ifdef TIMING
        auto elapsed_u = clock64() - start_u;
        if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
          printf("Solve U: %lld\n", elapsed_u);

#endif

        // store
        for (unsigned int i = 0; i < n_patch_dofs_rt / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_rt)
            {
              // unsigned int global_dof_index =
              //   gpu_data.patch_dof_smooth
              //     [patch * n_patch_dofs_all +
              //      gpu_data.htol_rt_interior[tid + i * block_size]];

              unsigned int global_dof_index =
                gpu_data.first_dof_rt
                  [patch * n_first_dof_rt +
                   gpu_data.base_dof_rt
                     [gpu_data.htol_rt_interior[tid + i * block_size]]] +
                gpu_data.dof_offset_rt
                  [gpu_data.htol_rt_interior[tid + i * block_size]];

              dst[global_dof_index] +=
                shared_data.local_dst[local_patch * n_patch_dofs + tid +
                                      i * block_size] *
                gpu_data.relaxation;
            }

        for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_dg)
            {
              // unsigned int global_dof_index =
              //   gpu_data
              //     .patch_dof_smooth[patch * n_patch_dofs_all +
              //                       n_patch_dofs_rt_all + tid + i *
              //                       block_size];

              unsigned int global_dof_index =
                gpu_data
                  .first_dof_dg[patch * n_first_dof_dg +
                                gpu_data.base_dof_dg[tid + i * block_size]] +
                gpu_data.dof_offset_dg[tid + i * block_size];

              dst[global_dof_index] +=
                shared_data.local_dst[local_patch * n_patch_dofs +
                                      n_patch_dofs_rt + tid + i * block_size] *
                gpu_data.relaxation;
            }
      }
  }

  template <int dim,
            int fe_degree,
            typename Number,
            LocalSolverVariant local_solver>
  __global__
    typename std::enable_if<local_solver == LocalSolverVariant::SchurIter>::type
    loop_kernel_global(
      const Number                                                 *src,
      Number                                                       *dst,
      const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {}

  template <int dim,
            int fe_degree,
            typename Number,
            LocalSolverVariant local_solver>
  __global__
    typename std::enable_if<local_solver != LocalSolverVariant::Direct &&
                            local_solver != LocalSolverVariant::SchurDirect &&
                            local_solver != LocalSolverVariant::SchurIter &&
                            local_solver !=
                              LocalSolverVariant::SchurTensorProduct>::type
    loop_kernel_global(
      const Number                                                 *src,
      Number                                                       *dst,
      const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    printf("Not Impl.!!!\n");
  }


  template <int dim,
            int fe_degree,
            typename Number,
            LaplaceVariant     laplace,
            LocalSolverVariant local_solver>
  __global__ void
  loop_kernel_fused_l(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr int n_dofs_1d = 2 * fe_degree + 3;
    constexpr int n_dofs_2d = n_dofs_1d * n_dofs_1d;

    constexpr int n_patch_dofs_rt =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 1);
    constexpr int n_patch_dofs_dg = Util::pow(2 * fe_degree + 2, dim);
    constexpr int n_patch_dofs    = n_patch_dofs_rt + n_patch_dofs_dg;
    constexpr int n_patch_dofs_rt_all =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 3);
    constexpr int n_patch_dofs_all =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * (fe_degree + 2) - 1) +
      Util::pow(2 * fe_degree + 2, dim);
    constexpr int block_size = n_dofs_2d * dim;

    constexpr int n_first_dof_rt = n_first_dofs_rt<dim>();
    constexpr int n_first_dof_dg = 1 << dim;

    const int patch_per_block = gpu_data.patch_per_block;
    const int local_patch     = threadIdx.y / (n_dofs_1d * dim);
    const int patch           = local_patch + patch_per_block * blockIdx.x;

    const int tid_y = threadIdx.y % (n_dofs_1d * dim);
    const int tid_x = threadIdx.x;
    const int tid   = tid_y * n_dofs_1d + tid_x;

    SharedDataSmoother<dim, Number, SmootherVariant::FUSED_L, local_solver>
      shared_data(get_shared_data_ptr<Number>(),
                  patch_per_block,
                  n_dofs_1d,
                  n_patch_dofs_all,
                  n_patch_dofs_dg);

    if (patch < gpu_data.n_patches)
      {
        // L M
        for (int dir = 0; dir < dim; ++dir)
          for (int d = 0; d < dim; ++d)
            if ((d == 0 && tid < n_dofs_2d) ||
                (d != 0 && tid < (n_dofs_1d - 1) * (n_dofs_1d - 1)))
              {
                auto dd = dir == 0 ? d :
                          dir == 1 ? (d == 0 ? 1 :
                                      d == 1 ? 0 :
                                               2) :
                                     (d == 0 ? 2 :
                                      d == 1 ? 0 :
                                               1);

                const int shift = local_patch * n_dofs_2d * dim;
                shared_data.local_mass[shift + d * n_dofs_2d + tid] =
                  gpu_data.smooth_mass_1d[d][tid];
                const int shift1 = local_patch * n_dofs_2d * dim * dim;
                shared_data
                  .local_laplace[shift1 + (dir * dim + d) * n_dofs_2d + tid] =
                  gpu_data
                    .smooth_stiff_1d[d][gpu_data.patch_type[patch * dim + dd] *
                                          n_dofs_2d +
                                        tid];
              }

        {
          // D
          const int shift1 = local_patch * n_dofs_2d * 1;
          if (tid < n_dofs_1d * (n_dofs_1d - 1))
            shared_data.local_mix_der[shift1 + tid] =
              -gpu_data.smooth_mixder_1d[0][tid];
        }

        for (unsigned int i = 0; i < n_patch_dofs_rt_all / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_rt_all)
            {
              // unsigned int global_dof_index =
              //   gpu_data
              //     .patch_dof_smooth[patch * n_patch_dofs_all +
              //                       gpu_data.htol_rt[tid + i * block_size]];

              unsigned int global_dof_index =
                gpu_data
                  .first_dof_rt[patch * n_first_dof_rt +
                                gpu_data.base_dof_rt
                                  [gpu_data.htol_rt[tid + i * block_size]]] +
                gpu_data.dof_offset_rt[gpu_data.htol_rt[tid + i * block_size]];

              shared_data.local_src[local_patch * n_patch_dofs_all + tid +
                                    i * block_size] = src[global_dof_index];
              shared_data.local_dst[local_patch * n_patch_dofs_all + tid +
                                    i * block_size] = dst[global_dof_index];
            }
        for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_dg)
            {
              // unsigned int global_dof_index_n =
              //   gpu_data
              //     .patch_dof_smooth[patch * n_patch_dofs_all +
              //                       n_patch_dofs_rt_all +
              //                       gpu_data.htol_dgn[tid + i * block_size]];

              unsigned int global_dof_index_n =
                gpu_data
                  .first_dof_dg[patch * n_first_dof_dg +
                                gpu_data.base_dof_dg
                                  [gpu_data.htol_dgn[tid + i * block_size]]] +
                gpu_data.dof_offset_dg[gpu_data.htol_dgn[tid + i * block_size]];

              shared_data
                .local_src[local_patch * n_patch_dofs_all +
                           n_patch_dofs_rt_all + tid + i * block_size] =
                src[global_dof_index_n];
              shared_data
                .local_dst[local_patch * n_patch_dofs_all +
                           n_patch_dofs_rt_all + tid + i * block_size] =
                dst[global_dof_index_n];
            }

        evaluate_residual<dim,
                          fe_degree,
                          Number,
                          laplace,
                          decltype(shared_data)>(local_patch, &shared_data);
        __syncthreads();

        // eigs
        unsigned int patch_type = 0;
        for (unsigned int d = 0; d < dim; ++d)
          patch_type += gpu_data.patch_type[patch * dim + d] * Util::pow(3, d);

        for (int dir = 0; dir < dim; ++dir)
          for (int d = 0; d < dim; ++d)
            if ((d == 0 && tid < (n_dofs_1d - 2) * (n_dofs_1d - 1)) ||
                (d != 0 && tid < (n_dofs_1d - 1) * (n_dofs_1d - 1)))
              {
                if ((d == 0 && tid < n_dofs_1d - 2) ||
                    (d != 0 && tid < n_dofs_1d - 1))
                  {
                    const int shift = local_patch * n_dofs_1d * dim * dim;
                    shared_data
                      .local_mass[shift + (dir * dim + d) * n_dofs_1d + tid] =
                      gpu_data.eigvals[dir][patch_type * n_dofs_1d * dim +
                                            d * n_dofs_1d + tid];
                  }
                const int shift2 = local_patch * n_dofs_2d * dim * dim;
                shared_data
                  .local_laplace[shift2 + (dir * dim + d) * n_dofs_2d + tid] =
                  gpu_data.eigvecs[dir][patch_type * n_dofs_2d * dim +
                                        d * n_dofs_2d + tid];
              }

        {
          // D
          const int shift1 = local_patch * n_dofs_2d * 1;
          if (tid < (n_dofs_1d - 2) * (n_dofs_1d - 1))
            shared_data.local_mix_der[shift1 + tid] =
              -gpu_data.smooth_mixmass_1d[0][tid];
        }

        for (unsigned int i = 0; i < n_patch_dofs_rt_all / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_rt_all)
            {
              shared_data.local_dst[local_patch * n_patch_dofs_all +
                                    gpu_data.htol_rt[tid + i * block_size]] =
                shared_data.local_src[local_patch * n_patch_dofs_all + tid +
                                      i * block_size];
            }
        for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_dg)
            {
              shared_data
                .local_dst[local_patch * n_patch_dofs_all +
                           n_patch_dofs_rt_all + tid + i * block_size] =
                shared_data
                  .local_src[local_patch * n_patch_dofs_all +
                             n_patch_dofs_rt_all + tid + i * block_size];
            }
        __syncthreads();
        for (unsigned int i = 0; i < n_patch_dofs_rt / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_rt)
            {
              shared_data
                .local_src[local_patch * n_patch_dofs + tid + i * block_size] =
                shared_data
                  .local_dst[local_patch * n_patch_dofs_all +
                             gpu_data.htol_rt_interior[tid + i * block_size]];
            }
        for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_dg)
            {
              shared_data.local_src[local_patch * n_patch_dofs +
                                    n_patch_dofs_rt + tid + i * block_size] =
                -shared_data
                   .local_dst[local_patch * n_patch_dofs_all +
                              n_patch_dofs_rt_all + tid + i * block_size];
            }
        __syncthreads();

        /// B^T M^-1 B P = B^T M^-1 F - G
        // B^T M^-1 F - G
        evaluate_smooth_p<dim, fe_degree, Number, decltype(shared_data)>(
          local_patch, &shared_data);
        __syncthreads();

        evaluate_smooth_cg<dim, fe_degree, Number, decltype(shared_data)>(
          local_patch, &shared_data);
        __syncthreads();

        for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_dg)
            {
              shared_data.local_dst[local_patch * n_patch_dofs +
                                    n_patch_dofs_rt + tid + i * block_size] =
                shared_data.local_src[local_patch * n_patch_dofs +
                                      ltoh_dgn[tid + i * block_size]];

              shared_data
                .local_src[local_patch * n_patch_dofs + n_patch_dofs_dg +
                           ltoh_dgt[tid + i * block_size]] =
                shared_data.local_src[local_patch * n_patch_dofs +
                                      ltoh_dgn[tid + i * block_size]];

              if constexpr (dim == 3)
                shared_data
                  .local_src[local_patch * n_patch_dofs + 2 * n_patch_dofs_dg +
                             ltoh_dgz[tid + i * block_size]] =
                  shared_data.local_src[local_patch * n_patch_dofs +
                                        ltoh_dgn[tid + i * block_size]];
            }

        /// M U = F - B P
        evaluate_smooth_u<dim, fe_degree, Number, decltype(shared_data)>(
          local_patch, &shared_data);
        __syncthreads();

        // store
        for (unsigned int i = 0; i < n_patch_dofs_rt / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_rt)
            {
              // unsigned int global_dof_index =
              //   gpu_data.patch_dof_smooth
              //     [patch * n_patch_dofs_all +
              //      gpu_data.htol_rt_interior[tid + i * block_size]];

              unsigned int global_dof_index =
                gpu_data.first_dof_rt
                  [patch * n_first_dof_rt +
                   gpu_data.base_dof_rt
                     [gpu_data.htol_rt_interior[tid + i * block_size]]] +
                gpu_data.dof_offset_rt
                  [gpu_data.htol_rt_interior[tid + i * block_size]];

              dst[global_dof_index] +=
                shared_data.local_dst[local_patch * n_patch_dofs + tid +
                                      i * block_size] *
                gpu_data.relaxation;
            }

        for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_dg)
            {
              // unsigned int global_dof_index =
              //   gpu_data
              //     .patch_dof_smooth[patch * n_patch_dofs_all +
              //                       n_patch_dofs_rt_all + tid + i *
              //                       block_size];

              unsigned int global_dof_index =
                gpu_data
                  .first_dof_dg[patch * n_first_dof_dg +
                                gpu_data.base_dof_dg[tid + i * block_size]] +
                gpu_data.dof_offset_dg[tid + i * block_size];

              dst[global_dof_index] +=
                shared_data.local_dst[local_patch * n_patch_dofs +
                                      n_patch_dofs_rt + tid + i * block_size] *
                gpu_data.relaxation;
            }
      }
  }

} // namespace PSMF

#endif // LOOP_KERNEL_CUH
