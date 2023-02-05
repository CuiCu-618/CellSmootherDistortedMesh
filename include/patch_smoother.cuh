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

extern double smoother_mem = 0;

namespace PSMF
{

  // Forward declaration
  template <typename MatrixType,
            int             dim,
            int             fe_degree,
            SmootherVariant kernel,
            DoFLayout       dof_layout>
  class PatchSmoother;

  /**
   * Implementation of vertex-patch precondition.
   */
  template <typename MatrixType,
            int             dim,
            int             fe_degree,
            SmootherVariant kernel,
            DoFLayout       dof_layout>
  class PatchSmootherImpl
  {
  public:
    using Number = typename MatrixType::value_type;
    using LevelVertexPatch =
      LevelVertexPatch<dim, fe_degree, Number, kernel, dof_layout>;
    using AdditionalData =
      typename PatchSmoother<MatrixType, dim, fe_degree, kernel, dof_layout>::
        AdditionalData;

    PatchSmootherImpl(const MatrixType     &A,
                      const AdditionalData &additional_data = AdditionalData());

    template <typename VectorType>
    void
    vmult(VectorType &dst, const VectorType &src) const;

    template <typename VectorType>
    void
    Tvmult(VectorType &, const VectorType &) const
    {}

    template <typename VectorType>
    void
    step(VectorType &dst, const VectorType &src) const;

    template <typename VectorType>
    void
    Tstep(VectorType &, const VectorType &) const
    {}

    std::size_t
    memory_consumption() const;

  private:
    template <typename VectorType>
    void
    step_impl(VectorType &dst, const VectorType &src) const;

    const SmartPointer<const MatrixType> A;
    LevelVertexPatch                     level_vertex_patch;

    const Number relaxation;
  };

  /**
   * Vertex-patch preconditioner.
   */
  template <typename MatrixType,
            int             dim,
            int             fe_degree,
            SmootherVariant kernel,
            DoFLayout       dof_layout>
  class PatchSmoother
    : public PreconditionRelaxation<
        MatrixType,
        PatchSmootherImpl<MatrixType, dim, fe_degree, kernel, dof_layout>>
  {
    using Number = typename MatrixType::value_type;
    using PreconditionerType =
      PatchSmootherImpl<MatrixType, dim, fe_degree, kernel, dof_layout>;

  public:
    class AdditionalData
    {
    public:
      AdditionalData(
        const Number            relaxation         = 1.,
        const bool              use_coloring       = true,
        const unsigned int      n_iterations       = 1,
        const unsigned int      patch_per_block    = 1,
        const GranularityScheme granularity_scheme = GranularityScheme::none);

      Number            relaxation;
      bool              use_coloring;
      unsigned int      n_iterations;
      unsigned int      patch_per_block;
      GranularityScheme granularity_scheme;
      /*
       * Preconditioner.
       */
      std::shared_ptr<PreconditionerType> preconditioner;
    };
    // using AdditionalData = typename BaseClass::AdditionalData;

    void
    initialize(const MatrixType     &A,
               const AdditionalData &parameters = AdditionalData());
  };

  /*--------------------- Implementation ------------------------*/

  template <typename MatrixType,
            int             dim,
            int             fe_degree,
            SmootherVariant kernel,
            DoFLayout       dof_layout>
  PatchSmootherImpl<MatrixType, dim, fe_degree, kernel, dof_layout>::
    PatchSmootherImpl(const MatrixType     &A,
                      const AdditionalData &additional_data_in)
    : A(&A)
    , relaxation(additional_data_in.relaxation)
  {
    typename LevelVertexPatch::AdditionalData additional_data;

    additional_data.relaxation         = additional_data_in.relaxation;
    additional_data.use_coloring       = additional_data_in.use_coloring;
    additional_data.patch_per_block    = additional_data_in.patch_per_block;
    additional_data.granularity_scheme = additional_data_in.granularity_scheme;

    level_vertex_patch.reinit(*(A.get_dof_handler()),
                              A.get_mg_level(),
                              additional_data);

    level_vertex_patch.reinit_tensor_product_smoother();
  }

  template <typename MatrixType,
            int             dim,
            int             fe_degree,
            SmootherVariant kernel,
            DoFLayout       dof_layout>
  template <typename VectorType>
  void
  PatchSmootherImpl<MatrixType, dim, fe_degree, kernel, dof_layout>::vmult(
    VectorType       &dst,
    const VectorType &src) const
  {
    dst = 0.;
    step(dst, src);
  }

  template <typename MatrixType,
            int             dim,
            int             fe_degree,
            SmootherVariant kernel,
            DoFLayout       dof_layout>
  template <typename VectorType>
  void
  PatchSmootherImpl<MatrixType, dim, fe_degree, kernel, dof_layout>::step(
    VectorType       &dst,
    const VectorType &src) const
  {
    Assert(this->A != nullptr, ExcNotInitialized());
    step_impl(dst, src);
  }

  template <typename MatrixType,
            int             dim,
            int             fe_degree,
            SmootherVariant kernel,
            DoFLayout       dof_layout>
  template <typename VectorType>
  void
  PatchSmootherImpl<MatrixType, dim, fe_degree, kernel, dof_layout>::step_impl(
    VectorType       &dst,
    const VectorType &src) const
  {
    unsigned int       level          = A->get_mg_level();
    unsigned int       n_dofs_per_dim = (1 << level) * (fe_degree + 2);
    // const unsigned int n_patches_1d   = (1 << level) - 1;

    switch (kernel)
      {
        case SmootherVariant::GLOBAL:
          {
            LocalSmoother_inverse<dim, fe_degree, Number, kernel, dof_layout>
              local_smoother_inverse(n_dofs_per_dim);
            level_vertex_patch.patch_loop_global(A,
                                                 local_smoother_inverse,
                                                 src,
                                                 dst);
            break;
          }
        case SmootherVariant::SEPERATE:
          {
            LocalSmoother<dim, fe_degree, Number, kernel, dof_layout>
              local_smoother(n_dofs_per_dim);

            LocalSmoother_inverse<dim, fe_degree, Number, kernel, dof_layout>
              local_smoother_inverse(n_dofs_per_dim);
            level_vertex_patch.patch_loop(local_smoother,
                                          src,
                                          dst,
                                          local_smoother_inverse);
            break;
          }
        case SmootherVariant::FUSED_BASE:
          {
            LocalSmoother<dim, fe_degree, Number, kernel, dof_layout>
              local_smoother(n_dofs_per_dim);

            level_vertex_patch.patch_loop(local_smoother, src, dst);
            break;
          }
        case SmootherVariant::FUSED_L:
          {
            LocalSmoother<dim, fe_degree, Number, kernel, dof_layout>
              local_smoother(n_dofs_per_dim);

            level_vertex_patch.patch_loop(local_smoother, src, dst);
            break;
          }
        case SmootherVariant::FUSED_3D:
          {
            LocalSmoother<dim, fe_degree, Number, kernel, dof_layout>
              local_smoother(n_dofs_per_dim);

            level_vertex_patch.patch_loop(local_smoother, src, dst);
            break;
          }
        case SmootherVariant::FUSED_CF:
          {
            LocalSmoother<dim, fe_degree, Number, kernel, dof_layout>
              local_smoother(n_dofs_per_dim);

            level_vertex_patch.patch_loop(local_smoother, src, dst);
            break;
          }
        default:
          AssertThrow(false, ExcMessage("Invalid Smoother Variant."));
          break;
      }
  }

  template <typename MatrixType,
            int             dim,
            int             fe_degree,
            SmootherVariant kernel,
            DoFLayout       dof_layout>
  std::size_t
  PatchSmootherImpl<MatrixType, dim, fe_degree, kernel, dof_layout>::
    memory_consumption() const
  {
    std::size_t result = sizeof(*this);
    result += level_vertex_patch.memory_consumption();
    return result;
  }

  template <typename MatrixType,
            int             dim,
            int             fe_degree,
            SmootherVariant kernel,
            DoFLayout       dof_layout>
  void
  PatchSmoother<MatrixType, dim, fe_degree, kernel, dof_layout>::initialize(
    const MatrixType                    &A,
    const PatchSmoother::AdditionalData &parameters_in)
  {
    Assert(parameters_in.preconditioner == nullptr, ExcInternalError());

    AdditionalData parameters;
    parameters.relaxation   = parameters_in.relaxation;
    parameters.n_iterations = parameters_in.n_iterations;
    parameters.preconditioner =
      std::make_shared<PreconditionerType>(A, parameters_in);

    // this->BaseClass::initialize(A, parameters);
    this->A          = &A;
    this->relaxation = parameters.relaxation;

    Assert(parameters.preconditioner, ExcNotInitialized());

    this->preconditioner = parameters.preconditioner;
    this->n_iterations   = parameters.n_iterations;
    smoother_mem += this->preconditioner->memory_consumption();
  }

  template <typename MatrixType,
            int             dim,
            int             fe_degree,
            SmootherVariant kernel,
            DoFLayout       dof_layout>
  inline PatchSmoother<MatrixType, dim, fe_degree, kernel, dof_layout>::
    AdditionalData::AdditionalData(const Number            relaxation,
                                   const bool              use_coloring,
                                   const unsigned int      n_iterations,
                                   const unsigned int      patch_per_block,
                                   const GranularityScheme granularity_scheme)
    : relaxation(relaxation)
    , use_coloring(use_coloring)
    , n_iterations(n_iterations)
    , patch_per_block(patch_per_block)
    , granularity_scheme(granularity_scheme)
  {}

} // namespace PSMF

#endif // PATCH_SMOOTHER_CUH