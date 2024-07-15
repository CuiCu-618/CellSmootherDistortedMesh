/**
 * @file cuda_mg_transfer.template.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief Implementation of the grid transfer operations.
 * @version 1.0
 * @date 2023-02-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef MG_TRANSFER_TEMPLATE_CUH
#define MG_TRANSFER_TEMPLATE_CUH

#include "cuda_mg_transfer.cuh"
#include "cuda_vector.cuh"

namespace PSMF
{

  enum TransferVariant
  {
    PROLONGATION,
    RESTRICTION
  };

  template <int dim, int fe_degree, typename Number>
  class MGTransferHelper
  {
  protected:
    static constexpr unsigned int n_coarse = fe_degree + 1;
    static constexpr unsigned int n_fine   = fe_degree * 2 + 2;
    static constexpr unsigned int M        = 2;

    Number                        *values;
    const Number                  *weights;
    const Number                  *shape_values;
    const types::global_dof_index *dof_indices_coarse;
    const types::global_dof_index *dof_indices_fine;

    __device__
    MGTransferHelper(Number                        *buf,
                     const Number                  *w,
                     const Number                  *shvals,
                     const types::global_dof_index *idx_coarse,
                     const types::global_dof_index *idx_fine)
      : values(buf)
      , weights(w)
      , shape_values(shvals)
      , dof_indices_coarse(idx_coarse)
      , dof_indices_fine(idx_fine)
    {}

    template <TransferVariant transfer_type, int dir>
    __device__ void
    reduce(const Number *my_shvals)
    {
      // multiplicity of large and small size
      constexpr bool         prol  = transfer_type == PROLONGATION;
      constexpr unsigned int n_src = prol ? n_coarse : n_fine;

      // in direction of reduction (dir and threadIdx.x respectively), always
      // read from 1 location, and write to M (typically 2). in other
      // directions, either read M or 1 and write same number.
      constexpr unsigned int M1 = prol ? M : 1;
      constexpr unsigned int M2 =
        prol ? (dir > 0 ? M : 1) : ((dir > 0 || dim < 2) ? 1 : M);
      constexpr unsigned int M3 =
        prol ? (dir > 1 ? M : 1) : ((dir > 1 || dim < 3) ? 1 : M);

      // const bool last_thread_x = 0;
      // const bool last_thread_y = 0;
      // const bool last_thread_z = 0;

      Number tmp[M1 * M2 * M3];

#pragma unroll
      for (int m3 = 0; m3 < M3; ++m3)
        {
#pragma unroll
          for (int m2 = 0; m2 < M2; ++m2)
            {
#pragma unroll
              for (int m1 = 0; m1 < M1; ++m1)
                {
                  tmp[m1 + M1 * (m2 + M2 * m3)] = 0;

                  for (int i = 0; i < n_src; ++i)
                    {
                      const unsigned int x = i;
                      const unsigned int y = m2 + M2 * threadIdx.y;
                      const unsigned int z = m3 + M3 * threadIdx.z;
                      const unsigned int idx =
                        (dir == 0 ? x + n_fine * (y + n_fine * z) :
                         dir == 1 ? y + n_fine * (x + n_fine * z) :
                                    y + n_fine * (z + n_fine * x));
                      // unless we are the last thread in a direction AND we
                      // are updating any value after the first one, go ahead
                      // if (((m1 == 0) || !last_thread_x) &&
                      //     ((m2 == 0) || !last_thread_y) &&
                      //     ((m3 == 0) || !last_thread_z))
                      {
                        tmp[m1 + M1 * (m2 + M2 * m3)] +=
                          my_shvals[m1 * n_src + i] * values[idx];
                      }
                    }
                }
            }
        }
      __syncthreads();

#pragma unroll
      for (int m3 = 0; m3 < M3; ++m3)
        {
#pragma unroll
          for (int m2 = 0; m2 < M2; ++m2)
            {
#pragma unroll
              for (int m1 = 0; m1 < M1; ++m1)
                {
                  const unsigned int x = m1 + M1 * threadIdx.x;
                  const unsigned int y = m2 + M2 * threadIdx.y;
                  const unsigned int z = m3 + M3 * threadIdx.z;
                  const unsigned int idx =
                    (dir == 0 ? x + n_fine * (y + n_fine * z) :
                     dir == 1 ? y + n_fine * (x + n_fine * z) :
                                y + n_fine * (z + n_fine * x));

                  // if (((m1 == 0) || !last_thread_x) &&
                  //     ((m2 == 0) || !last_thread_y) &&
                  //     ((m3 == 0) || !last_thread_z))
                  {
                    values[idx] = tmp[m1 + M1 * (m2 + M2 * m3)];
                  }
                }
            }
        }
    }
  };

  template <int dim, int fe_degree, typename Number>
  class MGProlongateHelper : public MGTransferHelper<dim, fe_degree, Number>
  {
    using MGTransferHelper<dim, fe_degree, Number>::M;
    using MGTransferHelper<dim, fe_degree, Number>::n_coarse;
    using MGTransferHelper<dim, fe_degree, Number>::n_fine;
    using MGTransferHelper<dim, fe_degree, Number>::dof_indices_coarse;
    using MGTransferHelper<dim, fe_degree, Number>::dof_indices_fine;
    using MGTransferHelper<dim, fe_degree, Number>::values;
    using MGTransferHelper<dim, fe_degree, Number>::shape_values;
    using MGTransferHelper<dim, fe_degree, Number>::weights;

  public:
    __device__
    MGProlongateHelper(Number                        *buf,
                       const Number                  *w,
                       const Number                  *shvals,
                       const types::global_dof_index *idx_coarse,
                       const types::global_dof_index *idx_fine)
      : MGTransferHelper<dim, fe_degree, Number>(buf,
                                                 w,
                                                 shvals,
                                                 idx_coarse,
                                                 idx_fine)
    {}

    __device__ void
    run(Number *dst, const Number *src)
    {
      Number my_shvals[M * n_coarse];
      for (int m = 0; m < (threadIdx.x < fe_degree ? M : M); ++m)
        for (int i = 0; i < n_coarse; ++i)
          my_shvals[m * n_coarse + i] =
            shape_values[(threadIdx.x * M + m) + n_fine * i];

      read_coarse(src);
      __syncthreads();

      this->template reduce<PROLONGATION, 0>(my_shvals);
      __syncthreads();
      if (dim > 1)
        {
          this->template reduce<PROLONGATION, 1>(my_shvals);
          __syncthreads();
          if (dim > 2)
            {
              this->template reduce<PROLONGATION, 2>(my_shvals);
              __syncthreads();
            }
        }

      // this->weigh_values();
      // __syncthreads();

      write_fine(dst);
    }

  private:
    __device__ void
    read_coarse(const Number *vec)
    {
      const unsigned int idx =
        threadIdx.x + n_fine * (threadIdx.y + n_fine * threadIdx.z);
      values[idx] = vec[dof_indices_coarse[idx]];
    }

    __device__ void
    write_fine(Number *vec) const
    {
      const unsigned int M1 = M;
      const unsigned int M2 = (dim > 1 ? M : 1);
      const unsigned int M3 = (dim > 2 ? M : 1);

      for (int m3 = 0; m3 < M3; ++m3)
        for (int m2 = 0; m2 < M2; ++m2)
          for (int m1 = 0; m1 < M1; ++m1)
            {
              const unsigned int x = (M1 * threadIdx.x + m1);
              const unsigned int y = (M2 * threadIdx.y + m2);
              const unsigned int z = (M3 * threadIdx.z + m3);

              const unsigned int idx = x + n_fine * (y + n_fine * z);
              if (x < n_fine && y < n_fine && z < n_fine)
                atomicAdd(&vec[dof_indices_fine[idx]], values[idx]);
            }
    }
  };

  template <int dim, int fe_degree, typename Number>
  class MGRestrictHelper : public MGTransferHelper<dim, fe_degree, Number>
  {
    using MGTransferHelper<dim, fe_degree, Number>::M;
    using MGTransferHelper<dim, fe_degree, Number>::n_coarse;
    using MGTransferHelper<dim, fe_degree, Number>::n_fine;
    using MGTransferHelper<dim, fe_degree, Number>::dof_indices_coarse;
    using MGTransferHelper<dim, fe_degree, Number>::dof_indices_fine;
    using MGTransferHelper<dim, fe_degree, Number>::values;
    using MGTransferHelper<dim, fe_degree, Number>::shape_values;
    using MGTransferHelper<dim, fe_degree, Number>::weights;

  public:
    __device__
    MGRestrictHelper(Number                        *buf,
                     const Number                  *w,
                     const Number                  *shvals,
                     const types::global_dof_index *idx_coarse,
                     const types::global_dof_index *idx_fine)
      : MGTransferHelper<dim, fe_degree, Number>(buf,
                                                 w,
                                                 shvals,
                                                 idx_coarse,
                                                 idx_fine)
    {}

    __device__ void
    run(Number *dst, const Number *src)
    {
      Number my_shvals[n_fine];
      for (int i = 0; i < n_fine; ++i)
        my_shvals[i] = shape_values[threadIdx.x * n_fine + i];

      read_fine(src);
      __syncthreads();
      // this->weigh_values();
      // __syncthreads();

      this->template reduce<RESTRICTION, 0>(my_shvals);
      __syncthreads();
      if (dim > 1)
        {
          this->template reduce<RESTRICTION, 1>(my_shvals);
          __syncthreads();
          if (dim > 2)
            {
              this->template reduce<RESTRICTION, 2>(my_shvals);
              __syncthreads();
            }
        }

      write_coarse(dst);
    }

  private:
    __device__ void
    read_fine(const Number *vec)
    {
      const unsigned int M1 = M;
      const unsigned int M2 = (dim > 1 ? M : 1);
      const unsigned int M3 = (dim > 2 ? M : 1);

      for (int m3 = 0; m3 < M3; ++m3)
        for (int m2 = 0; m2 < M2; ++m2)
          for (int m1 = 0; m1 < M1; ++m1)
            {
              const unsigned int x = (M1 * threadIdx.x + m1);
              const unsigned int y = (M2 * threadIdx.y + m2);
              const unsigned int z = (M3 * threadIdx.z + m3);

              const unsigned int idx = x + n_fine * (y + n_fine * z);
              if (x < n_fine && y < n_fine && z < n_fine)
                values[idx] = vec[dof_indices_fine[idx]];
            }
    }

    __device__ void
    write_coarse(Number *vec) const
    {
      const unsigned int idx =
        threadIdx.x + n_fine * (threadIdx.y + n_fine * threadIdx.z);

      // printf("[%d %d] ", idx, dof_indices_coarse[idx]);

      atomicAdd(&vec[dof_indices_coarse[idx]], values[idx]);
    }
  };

  namespace internal
  {
    extern __shared__ double shmem_d[];
    extern __shared__ float  shmem_f[];

    template <typename Number>
    __device__ inline Number *
    get_shared_mem_ptr();

    template <>
    __device__ inline double *
    get_shared_mem_ptr()
    {
      return shmem_d;
    }

    template <>
    __device__ inline float *
    get_shared_mem_ptr()
    {
      return shmem_f;
    }
  } // namespace internal

  template <int dim, int degree, typename loop_body, typename Number>
  __global__ void
  mg_kernel(Number                        *dst,
            const Number                  *src,
            const Number                  *weights,
            const Number                  *shape_values,
            const types::global_dof_index *dof_indices_coarse,
            const types::global_dof_index *dof_indices_fine,
            const types::global_dof_index *child_offset_in_parent,
            const unsigned int             n_child_cell_dofs)
  {
    const unsigned int            n_fine      = Util::pow(degree * 2 + 2, dim);
    const unsigned int            coarse_cell = blockIdx.x;
    const types::global_dof_index coarse_offset =
      child_offset_in_parent[coarse_cell];

    loop_body body(internal::get_shared_mem_ptr<Number>(),
                   weights + coarse_cell * Util::pow(3, dim),
                   shape_values,
                   dof_indices_coarse + coarse_offset,
                   dof_indices_fine + coarse_cell * n_child_cell_dofs);

    body.run(dst, src);
  }



  template <int dim, typename Number>
  template <template <int, int, typename> class loop_body, int degree>
  void
  MGTransferCUDA<dim, Number, DoFLayout::DGQ>::coarse_cell_loop(
    const unsigned int                                             fine_level,
    LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src)
    const
  {
    constexpr unsigned int n_fine_size =
      Util::pow(degree * 2 + 2, dim) * sizeof(Number);
    constexpr unsigned int n_coarse_dofs_1d = degree + 1;

    const unsigned int n_coarse_cells = n_owned_level_cells[fine_level - 1];

    // kernel parameters
    dim3 bk_dim(n_coarse_dofs_1d,
                (dim > 1) ? n_coarse_dofs_1d : 1,
                (dim > 2) ? n_coarse_dofs_1d : 1);

    dim3 gd_dim(n_coarse_cells);

    AssertCuda(cudaFuncSetAttribute(
      mg_kernel<dim, degree, loop_body<dim, degree, Number>, Number>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      n_fine_size));

    mg_kernel<dim, degree, loop_body<dim, degree, Number>>
      <<<gd_dim, bk_dim, n_fine_size>>>(
        dst.get_values(),
        src.get_values(),
        weights_on_refined[fine_level - 1]
          .get_values(), // only has fine-level entries
        prolongation_matrix_1d.get_values(),
        level_dof_indices[fine_level - 1].get_values(),
        level_dof_indices[fine_level].get_values(),
        child_offset_in_parent[fine_level - 1].get_values(), // on coarse level
        n_child_cell_dofs);

    AssertCudaKernel();
  }

  template <int dim, typename Number>
  MGTransferCUDA<dim, Number, DoFLayout::DGQ>::MGTransferCUDA()
    : fe_degree(0)
    , element_is_continuous(false)
    , n_components(0)
    , n_child_cell_dofs(0)
  {}

  template <int dim, typename Number>
  MGTransferCUDA<dim, Number, DoFLayout::DGQ>::MGTransferCUDA(
    const MGConstrainedDoFs &mg_c)
    : fe_degree(0)
    , element_is_continuous(false)
    , n_components(0)
    , n_child_cell_dofs(0)
  {
    this->mg_constrained_dofs = &mg_c;
  }

  template <int dim, typename Number>
  MGTransferCUDA<dim, Number, DoFLayout::DGQ>::~MGTransferCUDA()
  {}

  template <int dim, typename Number>
  void
  MGTransferCUDA<dim, Number, DoFLayout::DGQ>::initialize_constraints(
    const MGConstrainedDoFs &mg_c)
  {
    this->mg_constrained_dofs = &mg_c;
  }

  template <int dim, typename Number>
  void
  MGTransferCUDA<dim, Number, DoFLayout::DGQ>::clear()
  {
    fe_degree             = 0;
    element_is_continuous = false;
    n_components          = 0;
    n_child_cell_dofs     = 0;
    level_dof_indices.clear();
    child_offset_in_parent.clear();
    n_owned_level_cells.clear();
    weights_on_refined.clear();
  }

  template <int dim, typename Number>
  void
  MGTransferCUDA<dim, Number, DoFLayout::DGQ>::build(
    const DoFHandler<dim, dim> &mg_dof,
    const std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>
      &external_partitioners)
  {
    Assert(mg_dof.has_level_dofs(),
           ExcMessage(
             "The underlying DoFHandler object has not had its "
             "distribute_mg_dofs() function called, but this is a prerequisite "
             "for multigrid transfers. You will need to call this function, "
             "probably close to where you already call distribute_dofs()."));

    /**
     * Only global refinement so far, just plain copy. Uncomment for adaptice
     * refinement.
     */
    fill_copy_indices(mg_dof);

    const unsigned int n_levels = mg_dof.get_triangulation().n_global_levels();

    vector_partitioners.resize(0, n_levels - 1);
    for (unsigned int level = 0; level <= ghosted_level_vector.max_level();
         ++level)
      vector_partitioners[level] =
        ghosted_level_vector[level].get_partitioner();

    std::vector<std::vector<Number>>       weights_host;
    std::vector<std::vector<unsigned int>> level_dof_indices_host;
    std::vector<std::vector<std::pair<unsigned int, unsigned int>>>
      parent_child_connect;

    // std::vector<Table<2, unsigned int>> copy_indices_global_mine;
    // MGLevelObject<LinearAlgebra::distributed::Vector<Number>>
    //   ghosted_level_vector;
    std::vector<std::vector<std::vector<unsigned short>>>
      dirichlet_indices_host;

    ghosted_level_vector.resize(0, n_levels - 1);

    vector_partitioners.resize(0, n_levels - 1);
    for (unsigned int level = 0; level <= ghosted_level_vector.max_level();
         ++level)
      vector_partitioners[level] =
        ghosted_level_vector[level].get_partitioner();

    // WARN: setup_transfer() only works with "unsigned int"
    dealii::internal::MGTransfer::ElementInfo<Number> elem_info;
    dealii::internal::MGTransfer::setup_transfer<dim, Number>(
      mg_dof,
      this->mg_constrained_dofs,
      external_partitioners,
      elem_info,
      level_dof_indices_host,
      parent_child_connect,
      n_owned_level_cells,
      dirichlet_indices_host,
      weights_host,
      copy_indices_global_mine_host,
      vector_partitioners);

    // unpack element info data
    fe_degree             = elem_info.fe_degree;
    element_is_continuous = elem_info.element_is_continuous;
    n_components          = elem_info.n_components;
    n_child_cell_dofs     = elem_info.n_child_cell_dofs;

    //---------------------------------------------------------------------------
    // transfer stuff from host to device
    //---------------------------------------------------------------------------
    copy_to_device(prolongation_matrix_1d, elem_info.prolongation_matrix_1d);

    level_dof_indices.resize(n_levels);

    for (unsigned int l = 0; l < n_levels; l++)
      {
        copy_to_device(level_dof_indices[l], level_dof_indices_host[l]);
      }

    weights_on_refined.resize(n_levels - 1);
    // for (unsigned int l = 0; l < n_levels - 1; l++)
    //   {
    //     copy_to_device(weights_on_refined[l], weights_host[l]);
    //   }

    child_offset_in_parent.resize(n_levels - 1);
    std::vector<types::global_dof_index> offsets;

    for (unsigned int l = 0; l < n_levels - 1; l++)
      {
        offsets.resize(n_owned_level_cells[l]);

        for (unsigned int c = 0; c < n_owned_level_cells[l]; ++c)
          {
            const auto shift =
              dealii::internal::MGTransfer::compute_shift_within_children<dim>(
                parent_child_connect[l][c].second,
                fe_degree + 1 - element_is_continuous,
                fe_degree);
            offsets[c] =
              parent_child_connect[l][c].first * n_child_cell_dofs + shift;
          }

        copy_to_device(child_offset_in_parent[l], offsets);
      }

    std::vector<types::global_dof_index> dirichlet_index_vector;
    dirichlet_indices.resize(n_levels);
    if (this->mg_constrained_dofs != nullptr &&
        mg_constrained_dofs->have_boundary_indices())
      {
        for (unsigned int l = 0; l < n_levels; l++)
          {
            mg_constrained_dofs->get_boundary_indices(l).fill_index_vector(
              dirichlet_index_vector);
            copy_to_device(dirichlet_indices[l], dirichlet_index_vector);
          }
      }
  }

  template <int dim, typename Number>
  void
  MGTransferCUDA<dim, Number, DoFLayout::DGQ>::prolongate(
    const unsigned int                                             to_level,
    LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src)
    const
  {
    dst = 0;
    prolongate_and_add(to_level, dst, src);
  }

  template <int dim, typename Number>
  void
  MGTransferCUDA<dim, Number, DoFLayout::DGQ>::prolongate_and_add(
    const unsigned int                                             to_level,
    LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src)
    const
  {
    Assert((to_level >= 1) && (to_level <= level_dof_indices.size()),
           ExcIndexRange(to_level, 1, level_dof_indices.size() + 1));

    const bool src_inplace = src.get_partitioner().get() ==
                             this->vector_partitioners[to_level - 1].get();
    if (src_inplace == false)
      {
        if (this->ghosted_level_vector[to_level - 1].get_partitioner().get() !=
            this->vector_partitioners[to_level - 1].get())
          this->ghosted_level_vector[to_level - 1].reinit(
            this->vector_partitioners[to_level - 1]);
        this->ghosted_level_vector[to_level - 1].copy_locally_owned_data_from(
          src);
      }

    const bool dst_inplace =
      dst.get_partitioner().get() == this->vector_partitioners[to_level].get();
    if (dst_inplace == false)
      {
        if (this->ghosted_level_vector[to_level].get_partitioner().get() !=
            this->vector_partitioners[to_level].get())
          this->ghosted_level_vector[to_level].reinit(
            this->vector_partitioners[to_level]);
        AssertDimension(
          this->ghosted_level_vector[to_level].locally_owned_size(),
          dst.locally_owned_size());
        this->ghosted_level_vector[to_level] = 0.;
      }

    const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
      &src_vec = src_inplace ? src : this->ghosted_level_vector[to_level - 1];
    LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst_vec =
      dst_inplace ? dst : this->ghosted_level_vector[to_level];

    src_vec.update_ghost_values();

    if (fe_degree == 1)
      coarse_cell_loop<MGProlongateHelper, 1>(to_level, dst_vec, src_vec);
    else if (fe_degree == 2)
      coarse_cell_loop<MGProlongateHelper, 2>(to_level, dst_vec, src_vec);
    else if (fe_degree == 3)
      coarse_cell_loop<MGProlongateHelper, 3>(to_level, dst_vec, src_vec);
    else if (fe_degree == 4)
      coarse_cell_loop<MGProlongateHelper, 4>(to_level, dst_vec, src_vec);
    else if (fe_degree == 5)
      coarse_cell_loop<MGProlongateHelper, 5>(to_level, dst_vec, src_vec);
    else if (fe_degree == 6)
      coarse_cell_loop<MGProlongateHelper, 6>(to_level, dst_vec, src_vec);
    else if (fe_degree == 7)
      coarse_cell_loop<MGProlongateHelper, 7>(to_level, dst_vec, src_vec);
    else if (fe_degree == 8)
      coarse_cell_loop<MGProlongateHelper, 8>(to_level, dst_vec, src_vec);
    else if (fe_degree == 9)
      coarse_cell_loop<MGProlongateHelper, 9>(to_level, dst_vec, src_vec);
    else if (fe_degree == 10)
      coarse_cell_loop<MGProlongateHelper, 10>(to_level, dst_vec, src_vec);
    else
      AssertThrow(false,
                  ExcNotImplemented("Only degrees 1 through 10 implemented."));

    dst_vec.compress(VectorOperation::add);
    if (dst_inplace == false)
      dst += dst_vec;

    if (src_inplace == true)
      src.zero_out_ghost_values();
  }

  template <int dim, typename Number>
  void
  MGTransferCUDA<dim, Number, DoFLayout::DGQ>::restrict_and_add(
    const unsigned int                                             from_level,
    LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src)
    const
  {
    Assert((from_level >= 1) && (from_level <= level_dof_indices.size()),
           ExcIndexRange(from_level, 1, level_dof_indices.size() + 1));

    const bool src_inplace = src.get_partitioner().get() ==
                             this->vector_partitioners[from_level].get();
    if (src_inplace == false)
      {
        if (this->ghosted_level_vector[from_level].get_partitioner().get() !=
            this->vector_partitioners[from_level].get())
          this->ghosted_level_vector[from_level].reinit(
            this->vector_partitioners[from_level]);
        this->ghosted_level_vector[from_level].copy_locally_owned_data_from(
          src);
      }

    const bool dst_inplace = dst.get_partitioner().get() ==
                             this->vector_partitioners[from_level - 1].get();
    if (dst_inplace == false)
      {
        if (this->ghosted_level_vector[from_level - 1]
              .get_partitioner()
              .get() != this->vector_partitioners[from_level - 1].get())
          this->ghosted_level_vector[from_level - 1].reinit(
            this->vector_partitioners[from_level - 1]);
        AssertDimension(
          this->ghosted_level_vector[from_level - 1].locally_owned_size(),
          dst.locally_owned_size());
        this->ghosted_level_vector[from_level - 1] = 0.;
      }

    const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
      &src_vec = src_inplace ? src : this->ghosted_level_vector[from_level];
    LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst_vec =
      dst_inplace ? dst : this->ghosted_level_vector[from_level - 1];

    src_vec.update_ghost_values();

    if (fe_degree == 1)
      coarse_cell_loop<MGRestrictHelper, 1>(from_level, dst_vec, src_vec);
    else if (fe_degree == 2)
      coarse_cell_loop<MGRestrictHelper, 2>(from_level, dst_vec, src_vec);
    else if (fe_degree == 3)
      coarse_cell_loop<MGRestrictHelper, 3>(from_level, dst_vec, src_vec);
    else if (fe_degree == 4)
      coarse_cell_loop<MGRestrictHelper, 4>(from_level, dst_vec, src_vec);
    else if (fe_degree == 5)
      coarse_cell_loop<MGRestrictHelper, 5>(from_level, dst_vec, src_vec);
    else if (fe_degree == 6)
      coarse_cell_loop<MGRestrictHelper, 6>(from_level, dst_vec, src_vec);
    else if (fe_degree == 7)
      coarse_cell_loop<MGRestrictHelper, 7>(from_level, dst_vec, src_vec);
    else if (fe_degree == 8)
      coarse_cell_loop<MGRestrictHelper, 8>(from_level, dst_vec, src_vec);
    else if (fe_degree == 9)
      coarse_cell_loop<MGRestrictHelper, 9>(from_level, dst_vec, src_vec);
    else if (fe_degree == 10)
      coarse_cell_loop<MGRestrictHelper, 10>(from_level, dst_vec, src_vec);
    else
      AssertThrow(false,
                  ExcNotImplemented("Only degrees 1 through 10 implemented."));

    dst_vec.compress(VectorOperation::add);
    if (dst_inplace == false)
      dst += dst_vec;

    if (src_inplace == true)
      src.zero_out_ghost_values();
  }

  template <typename Number>
  __global__ void
  set_mg_constrained_dofs_kernel(Number                        *vec,
                                 const types::global_dof_index *indices,
                                 types::global_dof_index        len,
                                 Number                         val)
  {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
      {
        vec[indices[idx]] = val;
      }
  }

  template <int dim, typename Number>
  void
  MGTransferCUDA<dim, Number, DoFLayout::DGQ>::set_mg_constrained_dofs(
    LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &vec,
    unsigned int                                                   level,
    Number                                                         val) const
  {
    const types::global_dof_index len = dirichlet_indices[level].size();
    if (len > 0)
      {
        const unsigned int bksize  = 256;
        const unsigned int nblocks = (len - 1) / bksize + 1;
        dim3               bk_dim(bksize);
        dim3               gd_dim(nblocks);

        set_mg_constrained_dofs_kernel<<<gd_dim, bk_dim>>>(
          vec.get_values(), dirichlet_indices[level].get_values(), len, val);
        AssertCudaKernel();
      }
  }

  template <int dim, typename Number>
  template <int spacedim, typename Number2>
  void
  MGTransferCUDA<dim, Number, DoFLayout::DGQ>::copy_to_mg(
    const DoFHandler<dim, spacedim> &mg_dof,
    MGLevelObject<LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>>
                                                                         &dst,
    const LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA> &src)
    const
  {
    AssertIndexRange(dst.max_level(),
                     mg_dof.get_triangulation().n_global_levels());
    AssertIndexRange(dst.min_level(), dst.max_level() + 1);

    LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
         &this_ghosted_global_vector   = ghosted_global_vector;
    auto &this_copy_indices            = copy_indices;
    auto &this_copy_indices_level_mine = copy_indices_level_mine;


    for (unsigned int level = dst.min_level(); level <= dst.max_level();
         ++level)
      if (dst[level].size() != mg_dof.n_dofs(level) ||
          dst[level].locally_owned_size() !=
            mg_dof.locally_owned_mg_dofs(level).n_elements())
        {
          // In case a ghosted level vector has been initialized, we can
          // simply use that as a template for the vector partitioning. If
          // not, we resort to the locally owned range of the dof handler.
          if (level <= ghosted_level_vector.max_level() &&
              ghosted_level_vector[level].size() == mg_dof.n_dofs(level))
            dst[level].reinit(ghosted_level_vector[level], false);
          else
            dst[level].reinit(mg_dof.locally_owned_mg_dofs(level),
                              mg_dof.get_communicator());
        }
      else if ((perform_plain_copy == false &&
                perform_renumbered_plain_copy == false) ||
               level != dst.max_level())
        dst[level] = 0;

    if (perform_plain_copy)
      {
        // In this case, we can simply copy the local range.
        AssertDimension(dst[dst.max_level()].locally_owned_size(),
                        src.locally_owned_size());

        plain_copy<false>(dst[dst.max_level()], src);

        return;
      }
    else if (perform_renumbered_plain_copy)
      {
      }


    // std::cout << "Warning! Non-plain copy encourted! \n";

    // copy the source vector to the temporary vector that we hold for the
    // purpose of data exchange
    // this_ghosted_global_vector = src;
    plain_copy<false>(this_ghosted_global_vector, src);
    this_ghosted_global_vector.update_ghost_values();

    for (unsigned int level = dst.max_level() + 1; level != dst.min_level();)
      {
        --level;
        auto &dst_level = dst[level];

        copy_with_indices(dst_level,
                          this_ghosted_global_vector,
                          this_copy_indices[level].level_indices,
                          this_copy_indices[level].global_indices);

        copy_with_indices(dst_level,
                          this_ghosted_global_vector,
                          this_copy_indices_level_mine[level].level_indices,
                          this_copy_indices_level_mine[level].global_indices);

        dst_level.compress(VectorOperation::insert);
      }
  }

  template <int dim, typename Number>
  template <int spacedim, typename Number2>
  void
  MGTransferCUDA<dim, Number, DoFLayout::DGQ>::copy_from_mg(
    const DoFHandler<dim, spacedim>                                &mg_dof,
    LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA> &dst,
    const MGLevelObject<
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>> &src) const
  {
    (void)mg_dof;
    AssertIndexRange(src.max_level(),
                     mg_dof.get_triangulation().n_global_levels());
    AssertIndexRange(src.min_level(), src.max_level() + 1);

    if (perform_plain_copy)
      {
        AssertDimension(dst.locally_owned_size(),
                        src[src.max_level()].locally_owned_size());
        plain_copy<false>(dst, src[src.max_level()]);
        return;
      }
    else if (perform_renumbered_plain_copy)
      {
      }


    // std::cout << "Warning! Non-plain copy encourted! \n";

    dst = 0;
    for (unsigned int level = src.min_level(); level <= src.max_level();
         ++level)
      {
        // the ghosted vector should already have the correct local size (but
        // different parallel layout)
        if (ghosted_level_vector[level].size() > 0)
          AssertDimension(ghosted_level_vector[level].locally_owned_size(),
                          src[level].locally_owned_size());

        // the first time around, we copy the source vector to the temporary
        // vector that we hold for the purpose of data exchange
        LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
          &ghosted_vector = ghosted_level_vector[level];

        if (ghosted_level_vector[level].size() > 0)
          ghosted_vector = src[level];

        const auto ghosted_vector_ptr =
          (ghosted_level_vector[level].size() > 0) ? &ghosted_vector :
                                                     &src[level];

        ghosted_vector_ptr->update_ghost_values();

        copy_with_indices(dst,
                          *ghosted_vector_ptr,
                          copy_indices[level].global_indices,
                          copy_indices[level].level_indices);

        copy_with_indices(dst,
                          *ghosted_vector_ptr,
                          copy_indices_global_mine[level].global_indices,
                          copy_indices_global_mine[level].level_indices);
      }
    dst.compress(VectorOperation::insert);
  }

  template <int dim, typename Number>
  template <int spacedim, typename Number2>
  void
  MGTransferCUDA<dim, Number, DoFLayout::DGQ>::copy_from_mg_add(
    const DoFHandler<dim, spacedim>                                &mg_dof,
    LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA> &dst,
    const MGLevelObject<
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>> &src) const
  {
    (void)mg_dof;
    AssertIndexRange(src.max_level(),
                     mg_dof.get_triangulation().n_global_levels());
    AssertIndexRange(src.min_level(), src.max_level() + 1);

    dst.zero_out_ghost_values();
    for (unsigned int level = src.min_level(); level <= src.max_level();
         ++level)
      {
        // the ghosted vector should already have the correct local size (but
        // different parallel layout)
        AssertDimension(ghosted_level_vector[level].locally_owned_size(),
                        src[level].locally_owned_size());

        // the first time around, we copy the source vector to the temporary
        // vector that we hold for the purpose of data exchange
        LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
          &ghosted_vector = ghosted_level_vector[level];
        ghosted_vector    = src[level];
        ghosted_vector.update_ghost_values();

        copy_with_indices<Number2, Number, true>(
          dst,
          ghosted_vector,
          copy_indices[level].global_indices,
          copy_indices[level].level_indices);

        copy_with_indices<Number2, Number, true>(
          dst,
          ghosted_vector,
          copy_indices_global_mine[level].global_indices,
          copy_indices_global_mine[level].level_indices);
      }
    dst.compress(VectorOperation::add);

    std::cout << "Warning! Non-plain copy encourted! \n";
  }

  template <int dim, typename Number>
  std::size_t
  MGTransferCUDA<dim, Number, DoFLayout::DGQ>::memory_consumption() const
  {
    std::size_t memory = 0;
    memory += MemoryConsumption::memory_consumption(copy_indices);
    memory += MemoryConsumption::memory_consumption(level_dof_indices);
    memory += MemoryConsumption::memory_consumption(child_offset_in_parent);
    memory += MemoryConsumption::memory_consumption(n_owned_level_cells);
    memory += prolongation_matrix_1d.memory_consumption();
    memory += MemoryConsumption::memory_consumption(weights_on_refined);
    memory += MemoryConsumption::memory_consumption(dirichlet_indices);
    return memory;
  }

  template <int dim, typename Number>
  template <typename VectorType, typename VectorType2>
  void
  MGTransferCUDA<dim, Number, DoFLayout::DGQ>::copy_to_device(
    VectorType        &device,
    const VectorType2 &host)
  {
    LinearAlgebra::ReadWriteVector<typename VectorType::value_type> rw_vector(
      host.size());
    device.reinit(host.size());
    for (types::global_dof_index i = 0; i < host.size(); ++i)
      rw_vector[i] = host[i];
    device.import(rw_vector, VectorOperation::insert);
  }

  template <int dim, typename Number>
  void
  MGTransferCUDA<dim, Number, DoFLayout::DGQ>::fill_copy_indices(
    const DoFHandler<dim> &mg_dof)
  {
    const MPI_Comm mpi_communicator = mg_dof.get_communicator();

    // fill_internal
    std::vector<
      std::vector<std::pair<types::global_dof_index, types::global_dof_index>>>
      my_copy_indices;
    std::vector<
      std::vector<std::pair<types::global_dof_index, types::global_dof_index>>>
      my_copy_indices_global_mine;
    std::vector<
      std::vector<std::pair<types::global_dof_index, types::global_dof_index>>>
      my_copy_indices_level_mine;

    dealii::internal::MGTransfer::fill_copy_indices(mg_dof,
                                                    mg_constrained_dofs,
                                                    my_copy_indices,
                                                    my_copy_indices_global_mine,
                                                    my_copy_indices_level_mine);

    const unsigned int nlevels = mg_dof.get_triangulation().n_global_levels();

    IndexSet index_set(mg_dof.locally_owned_dofs().size());
    std::vector<types::global_dof_index> accessed_indices;
    ghosted_level_vector.resize(0, nlevels - 1);
    std::vector<IndexSet> level_index_set(nlevels);
    for (unsigned int l = 0; l < nlevels; ++l)
      {
        for (const auto &indices : my_copy_indices_level_mine[l])
          accessed_indices.push_back(indices.first);
        std::vector<types::global_dof_index> accessed_level_indices;
        for (const auto &indices : my_copy_indices_global_mine[l])
          accessed_level_indices.push_back(indices.second);
        std::sort(accessed_level_indices.begin(), accessed_level_indices.end());
        level_index_set[l].set_size(mg_dof.locally_owned_mg_dofs(l).size());
        level_index_set[l].add_indices(accessed_level_indices.begin(),
                                       accessed_level_indices.end());
        level_index_set[l].compress();
        ghosted_level_vector[l].reinit(mg_dof.locally_owned_mg_dofs(l),
                                       level_index_set[l],
                                       mpi_communicator);
      }
    std::sort(accessed_indices.begin(), accessed_indices.end());
    index_set.add_indices(accessed_indices.begin(), accessed_indices.end());
    index_set.compress();
    ghosted_global_vector.reinit(mg_dof.locally_owned_dofs(),
                                 index_set,
                                 mpi_communicator);

    // localize the copy indices for faster access. Since all access will be
    // through the ghosted vector in 'data', we can use this (much faster)
    // option
    copy_indices.resize(nlevels);
    copy_indices_level_mine.resize(nlevels);
    copy_indices_global_mine.resize(nlevels);
    copy_indices_global_mine_host.resize(nlevels);
    for (unsigned int level = 0; level < nlevels; ++level)
      {
        const Utilities::MPI::Partitioner &global_partitioner =
          *ghosted_global_vector.get_partitioner();
        const Utilities::MPI::Partitioner &level_partitioner =
          *ghosted_level_vector[level].get_partitioner();

        auto translate_indices =
          [&](const std::vector<
                std::pair<types::global_dof_index, types::global_dof_index>>
                           &global_copy_indices,
              IndexMapping &local_copy_indices) {
            const types::global_dof_index nmappings =
              global_copy_indices.size();
            std::vector<int> global_indices(nmappings);
            std::vector<int> level_indices(nmappings);

            for (types::global_dof_index j = 0; j < nmappings; ++j)
              {
                global_indices[j] = global_partitioner.global_to_local(
                  global_copy_indices[j].first);
                level_indices[j] = level_partitioner.global_to_local(
                  global_copy_indices[j].second);
              }

            copy_to_device(local_copy_indices.global_indices, global_indices);
            copy_to_device(local_copy_indices.level_indices, level_indices);
          };

        // owned-owned case
        translate_indices(my_copy_indices[level], copy_indices[level]);

        // remote-owned case
        translate_indices(my_copy_indices_level_mine[level],
                          copy_indices_level_mine[level]);

        // owned-remote case
        translate_indices(my_copy_indices_global_mine[level],
                          copy_indices_global_mine[level]);

        // copy_indices_global_mine_host
        copy_indices_global_mine_host[level].reinit(
          2, my_copy_indices_global_mine[level].size());
        for (types::global_dof_index i = 0;
             i < my_copy_indices_global_mine[level].size();
             ++i)
          {
            copy_indices_global_mine_host[level](0, i) =
              global_partitioner.global_to_local(
                my_copy_indices_global_mine[level][i].first);
            copy_indices_global_mine_host[level](1, i) =
              level_partitioner.global_to_local(
                my_copy_indices_global_mine[level][i].second);
          }
      }

    // Check if we can perform a cheaper "plain copy" (with or without
    // renumbering) instead of having to translate individual entries
    // using copy_indices*. This only works if a) we don't have to send
    // or receive any DoFs and we have all locally owned DoFs in our
    // copy_indices (so no adaptive refinement) and b) all processors
    // agree on the choice (see below).
    const bool my_perform_renumbered_plain_copy =
      (my_copy_indices.back().size() ==
       mg_dof.locally_owned_dofs().n_elements()) &&
      (my_copy_indices_global_mine.back().size() == 0) &&
      (my_copy_indices_level_mine.back().size() == 0);

    bool my_perform_plain_copy = false;
    if (my_perform_renumbered_plain_copy)
      {
        my_perform_plain_copy = true;
        // check whether there is a renumbering of degrees of freedom on
        // either the finest level or the global dofs, which means that we
        // cannot apply a plain copy
        for (types::global_dof_index i = 0; i < my_copy_indices.back().size();
             ++i)
          if (my_copy_indices.back()[i].first !=
              my_copy_indices.back()[i].second)
            {
              my_perform_plain_copy = false;
              break;
            }
      }

    // now do a global reduction over all processors to see what operation
    // they can agree upon
    perform_plain_copy =
      Utilities::MPI::min(static_cast<int>(my_perform_plain_copy),
                          mpi_communicator);
    perform_renumbered_plain_copy =
      Utilities::MPI::min(static_cast<int>(my_perform_renumbered_plain_copy),
                          mpi_communicator);

    // if we do a plain copy, no need to hold additional ghosted vectors
    if (perform_renumbered_plain_copy)
      {
      }
  }
} // namespace PSMF

#endif // MG_TRANSFER_TEMPLATE_CUH
