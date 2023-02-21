/**
 * @file laplace_operator.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief Implementation of the Laplace operations.
 * @version 1.0
 * @date 2023-02-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef LAPLACE_OPERATOR_CUH
#define LAPLACE_OPERATOR_CUH

#include "patch_base.cuh"

using namespace dealii;

namespace PSMF
{

  template <int dim, int fe_degree, typename Number, LaplaceVariant kernel>
  struct LocalLaplace
  {
    static constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;

    mutable std::size_t shared_mem;

    LocalLaplace()
      : shared_mem(0){};

    void
    setup_kernel(const unsigned int patch_per_block) const
    {
      constexpr unsigned int n =
        kernel == LaplaceVariant::ConflictFree ? 2 : (dim - 1);

      shared_mem = 0;

      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst
      shared_mem += 2 * patch_per_block * local_dim * sizeof(Number);
      // local_mass, local_derivative
      shared_mem +=
        2 * patch_per_block * n_dofs_1d * n_dofs_1d * 3 * sizeof(Number);
      // temp
      shared_mem += n * patch_per_block * local_dim * sizeof(Number);

      AssertCuda(cudaFuncSetAttribute(
        laplace_kernel_basic<dim, fe_degree, Number, kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem));
    }

    template <typename VectorType, typename DataType>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3       &block_dim) const
    {
      laplace_kernel_basic<dim, fe_degree, Number, kernel>
        <<<grid_dim, block_dim, shared_mem>>>(src.get_values(),
                                              dst.get_values(),
                                              gpu_data);
    }
  };

  template <int dim, int fe_degree, typename Number>
  struct LocalLaplace<dim, fe_degree, Number, LaplaceVariant::TensorCore>
  {
    static constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;

    mutable std::size_t shared_mem;

    LocalLaplace()
      : shared_mem(0){};

    void
    setup_kernel(const unsigned int patch_per_block) const
    {
      shared_mem = 0;

      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst
      shared_mem += 2 * patch_per_block * local_dim * sizeof(Number);
      // local_mass, local_derivative
      shared_mem +=
        2 * patch_per_block * n_dofs_1d * n_dofs_1d * 3 * sizeof(Number);
      // temp
      shared_mem += (dim - 2) * patch_per_block * local_dim * sizeof(Number);

      AssertCuda(
        cudaFuncSetAttribute(laplace_kernel_tensorcore<dim, fe_degree, Number>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shared_mem));
    }

    template <typename VectorType, typename DataType>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3       &block_dim) const
    {
      laplace_kernel_tensorcore<dim, fe_degree, Number>
        <<<grid_dim, block_dim, shared_mem>>>(src.get_values(),
                                              dst.get_values(),
                                              gpu_data);
    }
  };



  template <int dim, int fe_degree, typename Number, LaplaceVariant kernel>
  class LaplaceOperator : public Subscriptor
  {
  public:
    using value_type = Number;

    LaplaceOperator()
    {}

    void
    initialize(
      std::shared_ptr<const LevelVertexPatch<dim, fe_degree, Number>> data_,
      const DoFHandler<dim>                                          &mg_dof,
      const unsigned int                                              level)
    {
      data        = data_;
      dof_handler = &mg_dof;
      mg_level    = level;
    }

    void
    vmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
          const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
            &src) const
    {
      dst = 0.;

      LocalLaplace<dim, fe_degree, Number, kernel> local_laplace;

      data->cell_loop(local_laplace, src, dst);

      // mf_data->copy_constrained_values(src, dst);
    }

    void
    Tvmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
           const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
             &src) const
    {
      vmult(dst, src);
    }

    void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &vec) const
    {
      const unsigned int n_dofs = dof_handler->n_dofs(mg_level);
      vec.reinit(n_dofs);
    }

    void
    compute_residual(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src,
      const Function<dim, Number> &rhs_function,
      const Function<dim, Number> &exact_solution,
      const unsigned int           mg_level) const
    {
      (void)dst;
      (void)src;
      (void)rhs_function;
      (void)exact_solution;
      (void)mg_level;
    }

    unsigned int
    get_mg_level() const
    {
      return mg_level;
    }

    const DoFHandler<dim> *
    get_dof_handler() const
    {
      return dof_handler;
    }

    std::size_t
    memory_consumption() const
    {
      std::size_t result = sizeof(*this);
      return result;
    }

  private:
    std::shared_ptr<const LevelVertexPatch<dim, fe_degree, Number>> data;
    const DoFHandler<dim>                                          *dof_handler;
    unsigned int                                                    mg_level;
  };
} // namespace PSMF


#endif // LAPLACE_OPERATOR_CUH