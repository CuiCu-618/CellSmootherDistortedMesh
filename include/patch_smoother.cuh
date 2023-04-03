/**
 * Created by Cu Cui on 2022/12/25.
 */

#ifndef PATCH_SMOOTHER_CUH
#define PATCH_SMOOTHER_CUH

#include <deal.II/base/function.h>

#include <deal.II/lac/precondition.h>

#include "evaluate_kernel.cuh"
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
  struct LocalSmoother;


  template <typename MatrixType,
            int dim,
            int fe_degree,
            typename Number,
            LaplaceVariant lapalace>
  struct LocalSmoother<MatrixType,
                       dim,
                       fe_degree,
                       Number,
                       lapalace,
                       SmootherVariant::GLOBAL>
  {
    static constexpr unsigned int n_dofs_1d = 2 * fe_degree;

    LocalSmoother() = default;

    LocalSmoother(const SmartPointer<const MatrixType> A)
      : A(A)
    {
      A->initialize_dof_vector(tmp);
    }

    void
    setup_kernel(const unsigned int patch_per_block) const
    {
      shared_mem = 0;

      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst, local_residual
      shared_mem += 2 * patch_per_block * local_dim * sizeof(Number);
      // local_eigenvectors, local_eigenvalues
      shared_mem +=
        2 * patch_per_block * n_dofs_1d * n_dofs_1d * 1 * sizeof(Number);
      // temp
      shared_mem += (dim - 1) * patch_per_block * local_dim * sizeof(Number);

      AssertCuda(
        cudaFuncSetAttribute(loop_kernel_seperate_inv<dim, fe_degree, Number>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shared_mem));

      block_dim = dim3(patch_per_block * n_dofs_1d, n_dofs_1d);
    }

    template <typename VectorType, typename DataType>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                VectorType       &solution,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3 &,
                cudaStream_t stream) const
    {
      A->vmult(tmp, dst);
      tmp.sadd(-1., src);

      loop_kernel_seperate_inv<dim, fe_degree, Number>
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
  struct LocalSmoother<MatrixType,
                       dim,
                       fe_degree,
                       Number,
                       lapalace,
                       SmootherVariant::FUSED_L>
  {
  public:
    static constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;

    mutable std::size_t shared_mem;

    LocalSmoother() = default;

    LocalSmoother(const SmartPointer<const MatrixType>)
    {}

    void
    setup_kernel(const unsigned int patch_per_block) const
    {
      shared_mem = 0;

      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst
      shared_mem += 2 * patch_per_block * local_dim * sizeof(Number);
      // local_eigenvectors, local_eigenvalues
      shared_mem +=
        2 * patch_per_block * n_dofs_1d * n_dofs_1d * 1 * sizeof(Number);
      // temp
      shared_mem += (dim - 1) * patch_per_block * local_dim * sizeof(Number);

      AssertCuda(cudaFuncSetAttribute(
        loop_kernel_fused_l<dim, fe_degree, Number, lapalace>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem));
    }

    template <typename VectorType, typename DataType>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                VectorType       &solution,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3       &block_dim,
                cudaStream_t      stream) const
    {
      loop_kernel_fused_l<dim, fe_degree, Number, lapalace>
        <<<grid_dim, block_dim, shared_mem, stream>>>(src.get_values(),
                                                      dst.get_values(),
                                                      solution.get_values(),
                                                      gpu_data);
    }
  };


  template <typename MatrixType,
            int dim,
            int fe_degree,
            typename Number,
            LaplaceVariant lapalace>
  struct LocalSmoother<MatrixType,
                       dim,
                       fe_degree,
                       Number,
                       lapalace,
                       SmootherVariant::ConflictFree>
  {
  public:
    static constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;

    mutable std::size_t shared_mem;

    LocalSmoother() = default;

    LocalSmoother(const SmartPointer<const MatrixType>)
    {}

    void
    setup_kernel(const unsigned int patch_per_block) const
    {
      shared_mem = 0;

      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst
      shared_mem += 2 * patch_per_block * local_dim * sizeof(Number);
      // local_eigenvectors, local_eigenvalues
      shared_mem +=
        2 * patch_per_block * n_dofs_1d * n_dofs_1d * 1 * sizeof(Number);
      // temp
      shared_mem += 2 * patch_per_block * local_dim * sizeof(Number);

      AssertCuda(cudaFuncSetAttribute(
        loop_kernel_fused_cf<dim, fe_degree, Number, lapalace>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem));
    }

    template <typename VectorType, typename DataType>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                VectorType       &solution,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3       &block_dim,
                cudaStream_t      stream) const
    {
      loop_kernel_fused_cf<dim, fe_degree, Number, lapalace>
        <<<grid_dim, block_dim, shared_mem, stream>>>(src.get_values(),
                                                      dst.get_values(),
                                                      solution.get_values(),
                                                      gpu_data);
    }
  };



  // Forward declaration
  template <typename MatrixType,
            int             dim,
            int             fe_degree,
            LaplaceVariant  laplace,
            SmootherVariant smooth>
  class PatchSmoother;

  /**
   * Implementation of vertex-patch precondition.
   */
  template <typename MatrixType,
            int             dim,
            int             fe_degree,
            LaplaceVariant  laplace,
            SmootherVariant smooth>
  class PatchSmootherImpl
  {
  public:
    using Number = typename MatrixType::value_type;

    PatchSmootherImpl(
      const MatrixType                                               &A,
      std::shared_ptr<const LevelVertexPatch<dim, fe_degree, Number>> data_)
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
      LocalSmoother<MatrixType, dim, fe_degree, Number, laplace, smooth>
        local_smoother(A);

      data->patch_loop(local_smoother, src, dst);
    }

    template <typename VectorType>
    void
    Tstep(VectorType &, const VectorType &) const
    {}

    std::size_t
    memory_consumption() const
    {
      return sizeof(*this);
    }

  private:
    const SmartPointer<const MatrixType>                            A;
    std::shared_ptr<const LevelVertexPatch<dim, fe_degree, Number>> data;
  };

  /**
   * Vertex-patch preconditioner.
   */
  template <typename MatrixType,
            int             dim,
            int             fe_degree,
            LaplaceVariant  laplace,
            SmootherVariant smooth>
  class PatchSmoother
    : public PreconditionRelaxation<
        MatrixType,
        PatchSmootherImpl<MatrixType, dim, fe_degree, laplace, smooth>>
  {
    using Number = typename MatrixType::value_type;
    using PreconditionerType =
      PatchSmootherImpl<MatrixType, dim, fe_degree, laplace, smooth>;

  public:
    class AdditionalData
    {
    public:
      AdditionalData() = default;

      std::shared_ptr<LevelVertexPatch<dim, fe_degree, Number>> data;
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
  };

} // namespace PSMF

#endif // PATCH_SMOOTHER_CUH