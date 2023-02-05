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

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include "evaluate_kernel.cuh"
#include "patch_base.cuh"

using namespace dealii;

namespace PSMF
{

  template <int dim,
            int fe_degree,
            typename Number,
            SmootherVariant kernel,
            DoFLayout       dof_layout>
  class LaplaceOperator;

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  class LaplaceOperator<dim, fe_degree, Number, kernel, DoFLayout::DGQ>
    : public Subscriptor
  {
  public:
    using value_type = Number;
    using LevelVertexPatch =
      LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::DGQ>;
    using AdditionalData = typename LevelVertexPatch::AdditionalData;

    LaplaceOperator()
    {}

    void
    initialize(const DoFHandler<dim> &mg_dof,
               const unsigned int     level,
               const AdditionalData  &additional_data_in = AdditionalData())
    {
      dof_handler = &mg_dof;
      mg_level    = level;

      AdditionalData additional_data;

      additional_data.relaxation      = additional_data_in.relaxation;
      additional_data.use_coloring    = additional_data_in.use_coloring;
      additional_data.patch_per_block = additional_data_in.patch_per_block;
      additional_data.granularity_scheme =
        additional_data_in.granularity_scheme;

      level_vertex_patch.reinit(*dof_handler, mg_level, additional_data);
      level_vertex_patch.reinit_tensor_product_laplace();
    }

    void
    vmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
          const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
            &src) const
    {
      dst = 0.;

      const unsigned int n_dofs_per_dim = (1 << mg_level) * (fe_degree + 2);

      LocalLaplace<dim, fe_degree, Number, kernel, DoFLayout::DGQ>
        local_laplace(n_dofs_per_dim);

      level_vertex_patch.cell_loop(local_laplace, src, dst);
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
      dst = 0.;
      src.update_ghost_values();

      const unsigned int n_dofs = src.size();

      LinearAlgebra::distributed::Vector<Number, MemorySpace::Host>
        system_rhs_host(n_dofs);
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
        system_rhs_dev(n_dofs);

      LinearAlgebra::ReadWriteVector<Number> rw_vector(n_dofs);

      MappingQ1<dim>            mapping;
      AffineConstraints<Number> constraints;
      constraints.clear();
      DoFTools::make_hanging_node_constraints(*dof_handler, constraints);
      VectorTools::interpolate_boundary_values(
        mapping,
        *dof_handler,
        0,
        Functions::ZeroFunction<dim, Number>(),
        constraints);
      constraints.close();

      typename MatrixFree<dim, Number>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
        MatrixFree<dim, Number>::AdditionalData::none;
      additional_data.mapping_update_flags =
        (update_gradients | update_JxW_values | update_quadrature_points);
      additional_data.mapping_update_flags_inner_faces =
        (update_gradients | update_JxW_values | update_normal_vectors);
      additional_data.mapping_update_flags_boundary_faces =
        (update_gradients | update_JxW_values | update_normal_vectors |
         update_quadrature_points);
      MatrixFree<dim, Number> mf_data;
      mf_data.reinit(mapping,
                     *dof_handler,
                     constraints,
                     QGauss<1>(fe_degree + 1),
                     additional_data);
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(mf_data);

      for (unsigned int cell = 0; cell < mf_data.n_cell_batches(); ++cell)
        {
          phi.reinit(cell);
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              VectorizedArray<Number> rhs_val = VectorizedArray<Number>();
              Point<dim, VectorizedArray<Number>> point_batch =
                phi.quadrature_point(q);
              for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
                {
                  Point<dim> single_point;
                  for (unsigned int d = 0; d < dim; ++d)
                    single_point[d] = point_batch[d][v];
                  rhs_val[v] = rhs_function.value(single_point);
                }
              phi.submit_value(rhs_val, q);
            }
          phi.integrate_scatter(EvaluationFlags::values, system_rhs_host);
        }

      const Number penalty_factor = 1.0 * fe_degree * (fe_degree + 1);
      FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_face(
        mf_data, true);
      for (unsigned int face = mf_data.n_inner_face_batches();
           face <
           mf_data.n_inner_face_batches() + mf_data.n_boundary_face_batches();
           ++face)
        {
          phi_face.reinit(face);

          const VectorizedArray<Number> inverse_length_normal_to_face =
            std::abs((phi_face.get_normal_vector(0) *
                      phi_face.inverse_jacobian(0))[dim - 1]);
          const VectorizedArray<Number> sigma =
            inverse_length_normal_to_face * penalty_factor;

          for (unsigned int q = 0; q < phi_face.n_q_points; ++q)
            {
              VectorizedArray<Number> test_value = VectorizedArray<Number>(),
                                      test_normal_derivative =
                                        VectorizedArray<Number>();
              Point<dim, VectorizedArray<Number>> point_batch =
                phi_face.quadrature_point(q);

              for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
                {
                  Point<dim> single_point;
                  for (unsigned int d = 0; d < dim; ++d)
                    single_point[d] = point_batch[d][v];

                  if (mf_data.get_boundary_id(face) == 0)
                    test_value[v] = 2.0 * exact_solution.value(single_point);
                  else
                    {
                      Assert(false, ExcNotImplemented());
                      Tensor<1, dim> normal;
                      for (unsigned int d = 0; d < dim; ++d)
                        normal[d] = phi_face.get_normal_vector(q)[d][v];
                      test_normal_derivative[v] =
                        -normal * exact_solution.gradient(single_point);
                    }
                }
              phi_face.submit_value(test_value * sigma - test_normal_derivative,
                                    q);
              phi_face.submit_normal_derivative(-0.5 * test_value, q);
            }
          phi_face.integrate_scatter(EvaluationFlags::values |
                                       EvaluationFlags::gradients,
                                     system_rhs_host);
        }

      system_rhs_host.compress(VectorOperation::add);
      rw_vector.import(system_rhs_host, VectorOperation::insert);
      system_rhs_dev.import(rw_vector, VectorOperation::insert);

      vmult(dst, src);
      dst.sadd(-1., system_rhs_dev);
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
      result += level_vertex_patch.memory_consumption();
      return result;
    }

  private:
    const DoFHandler<dim> *dof_handler;
    LevelVertexPatch       level_vertex_patch;
    unsigned int           mg_level;
  };
} // namespace PSMF


#endif // LAPLACE_OPERATOR_CUH