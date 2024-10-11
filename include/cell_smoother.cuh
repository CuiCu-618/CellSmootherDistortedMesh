/**
 * Created by Cu Cui on 2022/12/25.
 */

#ifndef CELL_SMOOTHER_CUH
#define CELL_SMOOTHER_CUH

#include <deal.II/base/function.h>

#include <deal.II/lac/precondition.h>

#include "cell_base.cuh"
#include "cell_loop_kernel.cuh"
#include "patch_base.cuh"

using namespace dealii;

namespace PSMF
{
  template <typename MatrixType,
            int dim,
            int fe_degree,
            typename Number,
            LaplaceVariant  lapalace,
            SmootherVariant smooth>
  struct LocalCellSmoother
  {
  public:
    static constexpr unsigned int n_dofs_1d = fe_degree + 1;

    mutable std::size_t shared_mem;

    LocalCellSmoother() = default;

    LocalCellSmoother(const SmartPointer<const MatrixType>)
    {}

    template <bool is_ghost>
    void
    setup_kernel(const unsigned int) const
    {}

    template <typename VectorType, typename DataType, bool is_ghost>
    void
    loop_kernel(const VectorType &,
                VectorType &,
                VectorType &,
                const DataType &,
                const dim3 &,
                const dim3 &,
                cudaStream_t) const
    {}
  };


  template <typename MatrixType,
            int dim,
            int fe_degree,
            typename Number,
            LaplaceVariant lapalace>
  struct LocalCellSmoother<MatrixType,
                           dim,
                           fe_degree,
                           Number,
                           lapalace,
                           SmootherVariant::MCS>
  {
    static constexpr unsigned int n_dofs_1d = fe_degree + 1;

    LocalCellSmoother() = default;

    LocalCellSmoother(const SmartPointer<const MatrixType> A)
      : A(A)
    {
      A->initialize_dof_vector(tmp);
    }

    template <bool is_ghost>
    void
    setup_kernel(const unsigned int cell_per_block) const
    {
      shared_mem = 0;

      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst, local_residual
      shared_mem += 2 * cell_per_block * local_dim * sizeof(Number);
      // local_eigenvectors, local_eigenvalues
      shared_mem +=
        2 * cell_per_block * n_dofs_1d * n_dofs_1d * dim * sizeof(Number);
      // temp
      shared_mem += (dim - 1) * cell_per_block * local_dim * sizeof(Number);

      AssertCuda(cudaFuncSetAttribute(
        cell_loop_kernel_seperate_inv<dim, fe_degree, Number, is_ghost>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem));

      block_dim = dim3(cell_per_block * n_dofs_1d, n_dofs_1d);
    }

    template <typename VectorType, typename DataType, bool is_ghost>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                VectorType       &solution,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3 &,
                cudaStream_t stream) const
    {
      {
        A->vmult(tmp, dst);
        tmp.sadd(-1., src);

        // dst.update_ghost_values();
        // tmp.update_ghost_values();
      }

      if (grid_dim.x > 0)
        cell_loop_kernel_seperate_inv<dim, fe_degree, Number, is_ghost>
          <<<grid_dim, block_dim, shared_mem, stream>>>(tmp.get_values(),
                                                        dst.get_values(),
                                                        solution.get_values(),
                                                        gpu_data);
    }
    mutable std::size_t                  shared_mem;
    mutable dim3                         block_dim;
    const SmartPointer<const MatrixType> A;
    mutable LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> tmp;
  };


  template <typename MatrixType,
            int dim,
            int fe_degree,
            typename Number,
            LaplaceVariant lapalace>
  struct LocalCellSmoother<MatrixType,
                           dim,
                           fe_degree,
                           Number,
                           lapalace,
                           SmootherVariant::MCS_CG>
  {
    static constexpr unsigned int n_dofs_1d = fe_degree + 1;

    LocalCellSmoother() = default;

    LocalCellSmoother(const SmartPointer<const MatrixType> A)
      : A(A)
    {
      A->initialize_dof_vector(tmp);
    }

    template <bool is_ghost>
    void
    setup_kernel(const unsigned int cell_per_block) const
    {
      shared_mem = 0;

      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst, local_residual
      shared_mem += 2 * cell_per_block * local_dim * sizeof(Number);
      // local_eigenvectors, local_eigenvalues
      shared_mem +=
        2 * cell_per_block * n_dofs_1d * n_dofs_1d * dim * sizeof(Number);
      shared_mem += 6 * cell_per_block * n_dofs_1d * n_dofs_1d * sizeof(Number);
      // temp
      shared_mem += (dim + 3) * cell_per_block * local_dim * sizeof(Number);
      shared_mem += 7 * cell_per_block * local_dim * sizeof(Number);

      AssertCuda(cudaFuncSetAttribute(
        cell_loop_kernel_seperate_inv_cg<dim,
                                         fe_degree,
                                         Number,
                                         SmootherVariant::MCS_CG,
                                         is_ghost>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem));

      block_dim = dim3(cell_per_block * n_dofs_1d, n_dofs_1d);

      size_t newStackSize = 1024 * 3; // 2 KB
      AssertCuda(cudaDeviceSetLimit(cudaLimitStackSize, newStackSize));

      size_t stackSize;
      AssertCuda(cudaDeviceGetLimit(&stackSize, cudaLimitStackSize));

      // std::cout << "Current stack size per thread: " << stackSize << " bytes"
      //           << std::endl;
    }

    template <typename VectorType, typename DataType, bool is_ghost>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                VectorType       &solution,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3 &,
                cudaStream_t stream) const
    {
      {
        A->vmult(tmp, dst);
        tmp.sadd(-1., src);

        // dst.update_ghost_values();
        // tmp.update_ghost_values();
      }


      if (grid_dim.x > 0)
        {
          auto fe_data = A->get_mf_data()->get_cell_face_data(0);

          cell_loop_kernel_seperate_inv_cg<dim,
                                           fe_degree,
                                           Number,
                                           SmootherVariant::MCS_CG,
                                           is_ghost>
            <<<grid_dim, block_dim, shared_mem, stream>>>(tmp.get_values(),
                                                          dst.get_values(),
                                                          solution.get_values(),
                                                          gpu_data,
                                                          fe_data);
        }
    }
    mutable std::size_t                  shared_mem;
    mutable dim3                         block_dim;
    const SmartPointer<const MatrixType> A;
    mutable LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> tmp;
  };


  template <typename MatrixType,
            int dim,
            int fe_degree,
            typename Number,
            LaplaceVariant lapalace>
  struct LocalCellSmoother<MatrixType,
                           dim,
                           fe_degree,
                           Number,
                           lapalace,
                           SmootherVariant::MCS_PCG>
  {
    static constexpr unsigned int n_dofs_1d = fe_degree + 1;

    LocalCellSmoother() = default;

    LocalCellSmoother(const SmartPointer<const MatrixType> A)
      : A(A)
    {
      A->initialize_dof_vector(tmp);
    }

    template <bool is_ghost>
    void
    setup_kernel(const unsigned int cell_per_block) const
    {
      shared_mem = 0;

      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst, local_residual
      shared_mem += 2 * cell_per_block * local_dim * sizeof(Number);
      // local_eigenvectors, local_eigenvalues
      shared_mem +=
        2 * cell_per_block * n_dofs_1d * n_dofs_1d * dim * sizeof(Number);
      shared_mem += 6 * cell_per_block * n_dofs_1d * n_dofs_1d * sizeof(Number);
      // temp
      shared_mem += (dim + 4) * cell_per_block * local_dim * sizeof(Number);
      shared_mem += 7 * cell_per_block * local_dim * sizeof(Number);

      AssertCuda(cudaFuncSetAttribute(
        cell_loop_kernel_seperate_inv_cg<dim,
                                         fe_degree,
                                         Number,
                                         SmootherVariant::MCS_PCG,
                                         is_ghost>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem));

      block_dim = dim3(cell_per_block * n_dofs_1d, n_dofs_1d);

      size_t newStackSize = 1024 * 3; // 2 KB
      AssertCuda(cudaDeviceSetLimit(cudaLimitStackSize, newStackSize));

      size_t stackSize;
      AssertCuda(cudaDeviceGetLimit(&stackSize, cudaLimitStackSize));

      // std::cout << "Current stack size per thread: " << stackSize << " bytes"
      //           << std::endl;
    }

    template <typename VectorType, typename DataType, bool is_ghost>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                VectorType       &solution,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3 &,
                cudaStream_t stream) const
    {
      {
        A->vmult(tmp, dst);
        tmp.sadd(-1., src);

        // dst.update_ghost_values();
        // tmp.update_ghost_values();
      }

      if (grid_dim.x > 0)
        {
          auto fe_data = A->get_mf_data()->get_cell_face_data(0);

          cell_loop_kernel_seperate_inv_cg<dim,
                                           fe_degree,
                                           Number,
                                           SmootherVariant::MCS_PCG,
                                           is_ghost>
            <<<grid_dim, block_dim, shared_mem, stream>>>(tmp.get_values(),
                                                          dst.get_values(),
                                                          solution.get_values(),
                                                          gpu_data,
                                                          fe_data);
        }
    }
    mutable std::size_t                  shared_mem;
    mutable dim3                         block_dim;
    const SmartPointer<const MatrixType> A;
    mutable LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> tmp;
  };



  template <typename MatrixType,
            int dim,
            int fe_degree,
            typename Number,
            LaplaceVariant lapalace>
  struct LocalCellSmoother<MatrixType,
                           dim,
                           fe_degree,
                           Number,
                           lapalace,
                           SmootherVariant::FUSED_L>
  {
  public:
    static constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;

    mutable std::size_t shared_mem;

    LocalCellSmoother() = default;

    LocalCellSmoother(const SmartPointer<const MatrixType>)
    {}

    template <bool is_ghost>
    void
    setup_kernel(const unsigned int cell_per_block) const
    {
      shared_mem = 0;

      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst
      shared_mem += 2 * cell_per_block * local_dim * sizeof(Number);
      // local_eigenvectors, local_eigenvalues
      shared_mem +=
        2 * cell_per_block * n_dofs_1d * n_dofs_1d * 1 * sizeof(Number);
      // temp
      shared_mem += (dim - 1) * cell_per_block * local_dim * sizeof(Number);

      AssertCuda(cudaFuncSetAttribute(
        loop_kernel_fused_l<dim, fe_degree, Number, lapalace, is_ghost>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem));
    }

    template <typename VectorType, typename DataType, bool is_ghost>
    void
    loop_kernel(const VectorType &,
                VectorType &,
                VectorType &,
                const DataType &,
                const dim3 &,
                const dim3 &,
                cudaStream_t) const
    {
      // if (grid_dim.x > 0)
      //   loop_kernel_fused_l<dim, fe_degree, Number, lapalace, is_ghost>
      //     <<<grid_dim, block_dim, shared_mem, stream>>>(src.get_values(),
      //                                                   dst.get_values(),
      //                                                   solution.get_values(),
      //                                                   gpu_data);
    }
  };


  template <typename MatrixType,
            int dim,
            int fe_degree,
            typename Number,
            LaplaceVariant lapalace>
  struct LocalCellSmoother<MatrixType,
                           dim,
                           fe_degree,
                           Number,
                           lapalace,
                           SmootherVariant::ConflictFree>
  {
  public:
    static constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;

    mutable std::size_t shared_mem;

    LocalCellSmoother() = default;

    LocalCellSmoother(const SmartPointer<const MatrixType>)
    {}

    template <bool is_ghost>
    void
    setup_kernel(const unsigned int cell_per_block) const
    {
      shared_mem = 0;

      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst
      shared_mem += 2 * cell_per_block * local_dim * sizeof(Number);
      // local_eigenvectors, local_eigenvalues
      shared_mem +=
        2 * cell_per_block * n_dofs_1d * n_dofs_1d * 1 * sizeof(Number);
      // temp
      shared_mem += 2 * cell_per_block * local_dim * sizeof(Number);

      AssertCuda(cudaFuncSetAttribute(
        loop_kernel_fused_cf<dim, fe_degree, Number, lapalace, is_ghost>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem));
    }

    template <typename VectorType, typename DataType, bool is_ghost>
    void
    loop_kernel(const VectorType &,
                VectorType &,
                VectorType &,
                const DataType &,
                const dim3 &,
                const dim3 &,
                cudaStream_t) const
    {
      // if (grid_dim.x > 0)
      //   loop_kernel_fused_cf<dim, fe_degree, Number, lapalace, is_ghost>
      //     <<<grid_dim, block_dim, shared_mem, stream>>>(src.get_values(),
      //                                                   dst.get_values(),
      //                                                   solution.get_values(),
      //                                                   gpu_data);
    }
  };

  template <typename MatrixType,
            int dim,
            int fe_degree,
            typename Number,
            LaplaceVariant lapalace>
  struct LocalCellSmoother<MatrixType,
                           dim,
                           fe_degree,
                           Number,
                           lapalace,
                           SmootherVariant::ExactRes>
  {
  public:
    static constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;

    mutable std::size_t shared_mem;

    LocalCellSmoother() = default;

    LocalCellSmoother(const SmartPointer<const MatrixType>)
    {}

    template <bool is_ghost>
    void
    setup_kernel(const unsigned int cell_per_block) const
    {
      shared_mem = 0;

      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst
      shared_mem += 2 * cell_per_block * local_dim * sizeof(Number);
      // local_eigenvectors, local_eigenvalues
      shared_mem +=
        2 * cell_per_block * n_dofs_1d * n_dofs_1d * 1 * sizeof(Number);
      // temp
      shared_mem += 2 * cell_per_block * local_dim * sizeof(Number);

      AssertCuda(cudaFuncSetAttribute(
        loop_kernel_fused_exact<dim, fe_degree, Number, lapalace>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem));
    }

    template <typename VectorType, typename DataType, bool is_ghost>
    void
    loop_kernel(const VectorType &,
                VectorType &,
                VectorType &,
                const DataType &,
                const dim3 &,
                const dim3 &,
                cudaStream_t) const
    {
      // if (grid_dim.x > 0)
      //   loop_kernel_fused_exact<dim, fe_degree, Number, lapalace>
      //     <<<grid_dim, block_dim, shared_mem, stream>>>(src.get_values(),
      //                                                   dst.get_values(),
      //                                                   solution.get_values(),
      //                                                   gpu_data);
    }
  };


  // Forward declaration
  template <typename MatrixType,
            int             dim,
            int             fe_degree,
            LaplaceVariant  laplace,
            SmootherVariant smooth>
  class CellSmoother;

  /**
   * Implementation of vertex-patch precondition.
   */
  template <typename MatrixType,
            int             dim,
            int             fe_degree,
            LaplaceVariant  laplace,
            SmootherVariant smooth>
  class CellSmootherImpl
  {
  public:
    using Number = typename MatrixType::value_type;

    CellSmootherImpl(
      const MatrixType                                             &A,
      std::shared_ptr<const LevelCellPatch<dim, fe_degree, Number>> data_)
      : A(&A)
      , data(data_)
    {}

    template <typename VectorType>
    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      dst = 0.;
      step(dst, src);
    }

    template <typename VectorType>
    void
    Tvmult(VectorType &, const VectorType &) const
    {}

    template <typename VectorType>
    void
    step(VectorType &dst, const VectorType &src) const
    {
      LocalCellSmoother<MatrixType, dim, fe_degree, Number, laplace, smooth>
        local_smoother(A);

      data->cell_loop(local_smoother, src, dst);
    }

    template <typename VectorType>
    void
    Tstep(VectorType &, const VectorType &) const
    {}

    std::vector<double>
    get_cg_solver_info()
    {
      std::vector<double> vec;

      auto cg_data = data->get_cg_data();
      vec.push_back(cg_data[1] / cg_data[0]);
      vec.push_back(cg_data[2] / cg_data[0]);

      return vec;
    }

    std::size_t
    memory_consumption() const
    {
      return sizeof(*this);
    }

  private:
    const SmartPointer<const MatrixType>                          A;
    std::shared_ptr<const LevelCellPatch<dim, fe_degree, Number>> data;
  };

  /**
   * Vertex-patch preconditioner.
   */
  template <typename MatrixType,
            int             dim,
            int             fe_degree,
            LaplaceVariant  laplace,
            SmootherVariant smooth>
  class CellSmoother
    : public PreconditionRelaxation<
        MatrixType,
        CellSmootherImpl<MatrixType, dim, fe_degree, laplace, smooth>>
  {
    using Number = typename MatrixType::value_type;
    using PreconditionerType =
      CellSmootherImpl<MatrixType, dim, fe_degree, laplace, smooth>;

  public:
    class AdditionalData
    {
    public:
      AdditionalData() = default;

      std::shared_ptr<LevelCellPatch<dim, fe_degree, Number>> data;
      /*
       * Preconditioner.
       */
      std::shared_ptr<PreconditionerType> preconditioner;
    };
    // using AdditionalData = typename BaseClass::AdditionalData;

    void
    initialize(const MatrixType &A, const AdditionalData &parameters_in)
    {
      Assert(parameters_in.preconditioner == nullptr, ExcInternalError());

      AdditionalData parameters;
      parameters.preconditioner =
        std::make_shared<PreconditionerType>(A, parameters_in.data);

      this->A            = &A;
      this->relaxation   = 1;
      this->n_iterations = 1;

      Assert(parameters.preconditioner, ExcNotInitialized());
      this->preconditioner = parameters.preconditioner;
    }

    std::vector<double>
    get_cg_solver_info()
    {
      return this->preconditioner->get_cg_solver_info();
    }
  };

} // namespace PSMF

#endif // CELL_SMOOTHER_CUH
