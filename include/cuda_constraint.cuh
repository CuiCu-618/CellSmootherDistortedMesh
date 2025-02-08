/**
 * @file cuda_constraint.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief Constraint Handler
 * @version 1.0
 * @date 2023-03-10
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef CUDA_CONSTRAINT_CUH
#define CUDA_CONSTRAINT_CUH

#include <deal.II/base/config.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>

#include "cuda_vector.cuh"

namespace PSMF
{

  template <typename Number>
  __global__ void
  set_constrained_dofs_kernel(Number             *dst,
                              Number              val,
                              const unsigned int *constrained_dofs,
                              const unsigned int  n_constrained_dofs)
  {
    const unsigned int dof = threadIdx.x + blockDim.x * blockIdx.x;
    if (dof < n_constrained_dofs)
      {
        dst[constrained_dofs[dof]] = val;
      }
    __syncthreads();
  }

  template <typename Number>
  __global__ void
  save_constrained_dofs_kernel(Number             *in,
                               Number             *tmp_in,
                               const unsigned int *constrained_dofs,
                               const unsigned int  n_constrained_dofs)
  {
    const unsigned int dof = threadIdx.x + blockDim.x * blockIdx.x;
    if (dof < n_constrained_dofs)
      {
        tmp_in[dof]               = in[constrained_dofs[dof]];
        in[constrained_dofs[dof]] = 0;
      }
    __syncthreads();
  }


  template <typename Number>
  __global__ void
  save_constrained_dofs_kernel(const Number       *out,
                               Number             *in,
                               Number             *tmp_out,
                               Number             *tmp_in,
                               const unsigned int *constrained_dofs,
                               const unsigned int  n_constrained_dofs)
  {
    const unsigned int dof = threadIdx.x + blockDim.x * blockIdx.x;
    if (dof < n_constrained_dofs)
      {
        tmp_out[dof]              = out[constrained_dofs[dof]];
        tmp_in[dof]               = in[constrained_dofs[dof]];
        in[constrained_dofs[dof]] = 0;
      }
    __syncthreads();
  }

  template <typename Number>
  __global__ void
  load_constrained_dofs_kernel(Number             *in,
                               const Number       *tmp_in,
                               const unsigned int *constrained_dofs,
                               const unsigned int  n_constrained_dofs)
  {
    const unsigned int dof = threadIdx.x + blockDim.x * blockIdx.x;
    if (dof < n_constrained_dofs)
      {
        in[constrained_dofs[dof]] = tmp_in[dof];
      }
    __syncthreads();
  }


  template <typename Number>
  __global__ void
  load_and_add_constrained_dofs_kernel(Number             *out,
                                       Number             *in,
                                       const Number       *tmp_out,
                                       const Number       *tmp_in,
                                       const unsigned int *constrained_dofs,
                                       const unsigned int  n_constrained_dofs)
  {
    const unsigned int dof = threadIdx.x + blockDim.x * blockIdx.x;
    if (dof < n_constrained_dofs)
      {
        out[constrained_dofs[dof]] = tmp_out[dof] + tmp_in[dof];
        in[constrained_dofs[dof]]  = tmp_in[dof];
      }
    __syncthreads();
  }



  template <typename Number>
  class ConstraintHandler
  {
  public:
    // ConstraintHandler();

    using VectorType =
      dealii::LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>;

    void
    reinit(const dealii::AffineConstraints<Number> &constraints,
           const unsigned int                       n_dofs);

    void
    reinit(const dealii::MGConstrainedDoFs &mg_constrained_dofs,
           const unsigned int               level);


    void
    set_constrained_values(VectorType &v, Number val) const;

    void
    save_constrained_values(VectorType &v);

    void
    save_constrained_values(const VectorType &v1, VectorType &v2);

    void
    load_constrained_values(VectorType &v) const;

    void
    load_and_add_constrained_values(VectorType &v1, VectorType &v2) const;

    void
    copy_edge_values(VectorType &dst, const VectorType &src) const;

  private:
    void
    reinit_kernel_parameters();

    template <typename Number1>
    void
    alloc_and_copy(
      Number1 **array_device,
      const dealii::ArrayView<const Number1, dealii::MemorySpace::Host>
                         array_host,
      const unsigned int n);

    unsigned int n_constrained_dofs;
    unsigned int n_constrained_edge_dofs;

    // index lists
    unsigned int *constrained_indices;
    unsigned int *edge_indices;

    // temporary buffers
    VectorType constrained_values_src;
    VectorType constrained_values_dst;

    dim3 grid_dim;
    dim3 block_dim;
  };


  template <typename Number>
  void
  ConstraintHandler<Number>::reinit_kernel_parameters()
  {
    constexpr int block_size = 256;

    const unsigned int num_blocks = 1 + (n_constrained_dofs - 1) / block_size;

    grid_dim  = dim3(num_blocks);
    block_dim = dim3(block_size);
  }

  template <typename Number>
  template <typename Number1>
  void
  ConstraintHandler<Number>::alloc_and_copy(
    Number1 **array_device,
    const dealii::ArrayView<const Number1, dealii::MemorySpace::Host>
                       array_host,
    const unsigned int n)
  {
    cudaError_t error_code = cudaMalloc(array_device, n * sizeof(Number1));
    AssertCuda(error_code);
    AssertDimension(array_host.size(), n);

    error_code = cudaMemcpy(*array_device,
                            array_host.data(),
                            n * sizeof(Number1),
                            cudaMemcpyHostToDevice);
    AssertCuda(error_code);
  }

  template <typename Number>
  void
  ConstraintHandler<Number>::reinit(
    const dealii::AffineConstraints<Number> &constraints,
    const unsigned int                       n_dofs)
  {
    n_constrained_dofs = constraints.n_constraints();

    std::vector<unsigned int> constrained_dofs_host(n_constrained_dofs);

    unsigned int iconstr = 0;
    for (unsigned int i = 0; i < n_dofs; i++)
      {
        if (constraints.is_constrained(i))
          {
            constrained_dofs_host[iconstr] = i;
            iconstr++;
          }
      }

    // constrained_indices = constrained_dofs_host;
    alloc_and_copy(
      &constrained_indices,
      dealii::ArrayView<const unsigned int>(constrained_dofs_host.data(),
                                            constrained_dofs_host.size()),
      n_constrained_dofs);

    constrained_values_dst.reinit(n_constrained_dofs);
    constrained_values_src.reinit(n_constrained_dofs);

    // no edge constraints -- these are now hanging node constraints and are
    // handled by MatrixFreeGpu
    // edge_indices.clear();
    n_constrained_edge_dofs = 0;

    reinit_kernel_parameters();
  }


  template <typename Number>
  void
  ConstraintHandler<Number>::reinit(
    const dealii::MGConstrainedDoFs &mg_constrained_dofs,
    const unsigned int               level)
  {
    std::vector<types::global_dof_index> indices;
    IndexSet                             index_set;

    // first set up list of DoFs on refinement edges
    index_set = mg_constrained_dofs.get_refinement_edge_indices(level);
    index_set.fill_index_vector(indices);

    n_constrained_edge_dofs = indices.size();
    // edge_indices = indices;
    alloc_and_copy(&edge_indices,
                   dealii::ArrayView<const unsigned int>(indices.data(),
                                                         indices.size()),
                   n_constrained_edge_dofs);

    // then add also boundary DoFs to get all constrained DoFs
    index_set.add_indices(mg_constrained_dofs.get_boundary_indices(level));
    index_set.fill_index_vector(indices);

    n_constrained_dofs = indices.size();

    // constrained_indices = indices;
    alloc_and_copy(&constrained_indices,
                   dealii::ArrayView<const unsigned int>(indices.data(),
                                                         indices.size()),
                   n_constrained_dofs);

    constrained_values_dst.reinit(n_constrained_dofs);
    constrained_values_src.reinit(n_constrained_dofs);

    reinit_kernel_parameters();
  }


  template <typename Number>
  void
  ConstraintHandler<Number>::set_constrained_values(VectorType &dst,
                                                    Number      val) const
  {
    if (n_constrained_dofs != 0)
      {
        set_constrained_dofs_kernel<Number><<<grid_dim, block_dim>>>(
          dst.get_values(), val, constrained_indices, n_constrained_dofs);
        AssertCudaKernel();
      }
  }

  template <typename Number>
  void
  ConstraintHandler<Number>::save_constrained_values(VectorType &src)
  {
    if (n_constrained_dofs != 0)
      {
        save_constrained_dofs_kernel<Number>
          <<<grid_dim, block_dim>>>(src.get_values(),
                                    constrained_values_src.get_values(),
                                    constrained_indices,
                                    n_constrained_dofs);
        AssertCudaKernel();
      }
  }


  template <typename Number>
  void
  ConstraintHandler<Number>::save_constrained_values(const VectorType &dst,
                                                     VectorType       &src)
  {
    if (n_constrained_dofs != 0)
      {
        save_constrained_dofs_kernel<Number>
          <<<grid_dim, block_dim>>>(dst,
                                    src.get_values(),
                                    constrained_values_dst.get_values(),
                                    constrained_values_src.get_values(),
                                    constrained_indices,
                                    n_constrained_dofs);
        AssertCudaKernel();
      }
  }

  template <typename Number>
  void
  ConstraintHandler<Number>::load_constrained_values(VectorType &src) const
  {
    if (n_constrained_dofs != 0)
      {
        load_constrained_dofs_kernel<Number>
          <<<grid_dim, block_dim>>>(src.get_values(),
                                    constrained_values_src.get_values(),
                                    constrained_indices,
                                    n_constrained_dofs);
        AssertCudaKernel();
      }
  }


  template <typename Number>
  void
  ConstraintHandler<Number>::load_and_add_constrained_values(
    VectorType &dst,
    VectorType &src) const
  {
    if (n_constrained_dofs != 0)
      {
        load_and_add_constrained_dofs_kernel<Number>
          <<<grid_dim, block_dim>>>(dst.get_values(),
                                    src.get_values(),
                                    constrained_values_dst.get_values(),
                                    constrained_values_src.get_values(),
                                    constrained_indices,
                                    n_constrained_dofs);
        AssertCudaKernel();
      }
  }

  template <typename Number>
  void
  ConstraintHandler<Number>::copy_edge_values(VectorType       &dst,
                                              const VectorType &src) const
  {
    copy_with_indices(
      dst, src, edge_indices, edge_indices, n_constrained_edge_dofs);
  }

  // todo: bug Can't constrain a degree of freedom to itself
  template <int dim, int fe_degree, typename Number>
  class LevelAffineConstraints
  {
  public:
    LevelAffineConstraints()
    {}

    void
    reinit(const dealii::DoFHandler<dim>           &dof_handler,
           const dealii::AffineConstraints<Number> &constraints,
           const unsigned int                       max_level)
    {
      n_level_dofs.resize(max_level);
      level_constraints.resize(max_level);
      level_constraints_indicator.resize(max_level);
      level_constraints_weights.resize(max_level);


      std::vector<unsigned int> h_c;
      std::vector<int>          h_c_i;
      std::vector<double>       h_c_w;

      auto lines = constraints.get_lines();
      for (auto &l : lines)
        if (l.entries.size() > 0)
          for (auto e : l.entries)
            {
              h_c.push_back(l.index);
              h_c_i.push_back(e.first);
              h_c_w.push_back(e.second);
            }
        else
          {
            // h_c.push_back(l.index);
            // h_c_i.push_back(0);
            // h_c_w.push_back(0);
          }


      unsigned int global_counter = 0;
      unsigned int n_face_dofs    = std::pow(fe_degree + 1, dim - 1);

      std::vector<unsigned int> local_dof_indices_fine(
        dof_handler.get_fe().n_dofs_per_cell());
      std::vector<unsigned int> local_dof_indices_coarse(
        dof_handler.get_fe().n_dofs_per_cell());

      for (unsigned int level = 1; level < max_level; ++level)
        {
          n_level_dofs[level] = dof_handler.n_dofs(level);

          auto beginc = dof_handler.begin_mg(level);
          auto endc   = dof_handler.end_mg(level);

          for (auto cell = beginc; cell != endc; ++cell)
            for (const unsigned int face_no : cell->face_indices())
              if (!cell->at_boundary(face_no))
                {
                  auto neighbor = cell->neighbor_or_periodic_neighbor(face_no);
                  auto neighbor_face_no = cell->neighbor_face_no(face_no);

                  if (cell->neighbor_is_coarser(face_no))
                    {
                      cell->get_active_or_mg_dof_indices(
                        local_dof_indices_fine);
                      neighbor->get_active_or_mg_dof_indices(
                        local_dof_indices_coarse);

                      for (unsigned int i = 0; i < n_face_dofs; ++i)
                        {
                          auto g_h_c = h_c[global_counter];
                          auto g_h_i = h_c_i[global_counter];

                          auto fine_ind =
                            local_dof_indices_fine[face_no * n_face_dofs + i];
                          auto coarse_ind =
                            local_dof_indices_coarse[neighbor_face_no *
                                                       n_face_dofs +
                                                     i];

                          level_constraints[level].push_back(fine_ind);
                          level_constraints_indicator[level].push_back(
                            coarse_ind);
                          level_constraints_weights[level].push_back(
                            h_c_w[global_counter]);

                          global_counter++;
                          for (unsigned int j = 1; j < n_face_dofs; ++j)
                            if (g_h_c == h_c[global_counter])
                              {
                                level_constraints[level].push_back(fine_ind);
                                level_constraints_indicator[level].push_back(
                                  coarse_ind + h_c_i[global_counter] - g_h_i);
                                level_constraints_weights[level].push_back(
                                  h_c_w[global_counter]);

                                global_counter++;
                              }
                        }
                    }
                }
        }
    }

    const dealii::AffineConstraints<Number>
    get_level_constraints(const unsigned int level)
    {
      dealii::AffineConstraints<Number> level_constraint;

      for (unsigned int i = 0; i < level_constraints[level].size(); ++i)
        {
          level_constraint.add_line(level_constraints[level][i]);
          level_constraint.add_entry(level_constraints[level][i],
                                     level_constraints_indicator[level][i],
                                     level_constraints_weights[level][i]);
        }
      level_constraint.close();

      return level_constraint;
    }

    const dealii::AffineConstraints<Number>
    get_level_coarse_constraints(const unsigned int level)
    {
      std::set<unsigned int> s(level_constraints_indicator[level].begin(),
                               level_constraints_indicator[level].end());
      level_constraints_indicator[level].assign(s.begin(), s.end());

      dealii::AffineConstraints<Number> level_constraint;

      for (unsigned int i = 0; i < level_constraints_indicator[level].size();
           ++i)
        {
          level_constraint.add_line(n_level_dofs[level]);
          level_constraint.add_entry(n_level_dofs[level],
                                     level_constraints_indicator[level][i],
                                     -101);
        }

      level_constraint.close();

      return level_constraint;
    }

    const dealii::AffineConstraints<Number>
    get_level_plain_coarse_constraints(const unsigned int level)
    {
      std::set<unsigned int> s(level_constraints_indicator[level].begin(),
                               level_constraints_indicator[level].end());
      level_constraints_indicator[level].assign(s.begin(), s.end());

      dealii::AffineConstraints<Number> level_constraint;

      for (unsigned int i = 0; i < level_constraints_indicator[level].size();
           ++i)
        level_constraint.add_line(level_constraints_indicator[level][i]);


      level_constraint.close();

      return level_constraint;
    }

    std::vector<unsigned int>              n_level_dofs;
    std::vector<std::vector<unsigned int>> level_constraints;
    std::vector<std::vector<int>>          level_constraints_indicator;
    std::vector<std::vector<Number>>       level_constraints_weights;
  };


} // namespace PSMF

#endif // CUDA_CONSTRAINT_CUH
