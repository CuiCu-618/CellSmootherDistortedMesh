/**
 * Created by Cu Cui on 2022/12/25.
 */

#ifndef LAPLACE_OPERATOR_CUH
#define LAPLACE_OPERATOR_CUH

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include "dealii/cuda_fee.cuh"
#include "dealii/cuda_mf.cuh"

using namespace dealii;

namespace PSMF
{

  template <int dim, int fe_degree, typename Number>
  class LaplaceOperatorQuad
  {
  public:
    __device__
    LaplaceOperatorQuad()
    {}
    __device__ void
    operator()(
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> *fe_eval) const
    {
      fe_eval->submit_gradient(fe_eval->get_gradient());
    }
  };

  template <int dim, int fe_degree, typename Number>
  class LocalLaplaceOperator
  {
  public:
    static const unsigned int n_dofs_1d    = fe_degree + 1;
    static const unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
    static const unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);

    const unsigned int dofs_per_dim;

    LocalLaplaceOperator(const unsigned int dofs_per_dim)
      : dofs_per_dim(dofs_per_dim)
    {}

    __device__ void
    operator()(const unsigned int                            cell,
               const typename MatrixFree<dim, Number>::Data *gpu_data,
               SharedDataCuda<dim, Number>                  *shared_data,
               const Number                                 *src,
               Number                                       *dst) const
    {
      const unsigned int pos =
        local_q_point_id<dim, Number>(cell, gpu_data, n_dofs_1d, n_q_points);
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> fe_eval(
        cell, gpu_data, shared_data, dofs_per_dim);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(false, true);
      fe_eval.apply_for_each_quad_point(
        LaplaceOperatorQuad<dim, fe_degree, Number>());
      fe_eval.integrate(false, true);
      fe_eval.distribute_local_to_global(dst);
    }
  };

  template <int dim, int fe_degree, typename Number, DoFLayout dof_layout>
  class LaplaceOperator;

  template <int dim, int fe_degree, typename Number>
  class LaplaceOperator<dim, fe_degree, Number, DoFLayout::Q>
    : public Subscriptor
  {
  public:
    using value_type = Number;

    LaplaceOperator()
    {}

    void
    initialize(std::shared_ptr<const PSMF::MatrixFree<dim, Number>> data_)
    {
      mf_data = data_;
    }

    void
    vmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
          const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
            &src) const
    {
      dst = 0.;

      unsigned int level          = mf_data->get_mg_level();
      unsigned int n_dofs_per_dim = (1 << level) * fe_degree + 1;

      LocalLaplaceOperator<dim, fe_degree, Number> Laplace_operator(
        n_dofs_per_dim);

      mf_data->cell_loop(Laplace_operator, src, dst);
      mf_data->copy_constrained_values(src, dst);
    }

    void
    Tvmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
           const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
             &src) const
    {
      dst = 0.;
      LocalLaplaceOperator<dim, fe_degree, Number> Laplace_operator;
      mf_data->cell_loop(Laplace_operator, src, dst);
      mf_data->copy_constrained_values(src, dst);
    }

    void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &vec) const
    {
      mf_data->initialize_dof_vector(vec);
    }

    void
    compute_residual(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src,
      const Function<dim, Number> &rhs_function,
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

      AffineConstraints<Number> constraints;
      constraints.close();

      const QGauss<dim>  quadrature_formula(fe_degree + 1);
      FEValues<dim>      fe_values(mf_data->get_dof_handler().get_fe(),
                              quadrature_formula,
                              update_values | update_quadrature_points |
                                update_JxW_values);
      const unsigned int dofs_per_cell =
        mf_data->get_dof_handler().get_fe().n_dofs_per_cell();
      const unsigned int        n_q_points = quadrature_formula.size();
      Vector<Number>            cell_rhs(dofs_per_cell);
      std::vector<unsigned int> local_dof_indices(dofs_per_cell);
      std::vector<Number>       rhs_values(n_q_points);

      auto begin = mf_data->get_dof_handler().begin_mg(mg_level);
      auto end   = mf_data->get_dof_handler().end_mg(mg_level);

      for (auto cell = begin; cell != end; ++cell)
        if (cell->is_locally_owned_on_level())
          {
            cell_rhs = 0;
            fe_values.reinit(cell);
            rhs_function.value_list(fe_values.get_quadrature_points(),
                                    rhs_values);

            for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
              {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  cell_rhs(i) += (fe_values.shape_value(i, q_index) *
                                  rhs_values[q_index] * fe_values.JxW(q_index));
              }
            cell->get_mg_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(cell_rhs,
                                                   local_dof_indices,
                                                   system_rhs_host);
          }

      system_rhs_host.compress(VectorOperation::add);
      rw_vector.import(system_rhs_host, VectorOperation::insert);
      system_rhs_dev.import(rw_vector, VectorOperation::insert);

      vmult(dst, src);
      dst.sadd(-1., system_rhs_dev);
    }

    void
    clear()
    {
      mf_data.reset();
    }

    std::shared_ptr<const PSMF::MatrixFree<dim, Number>>
    get_matrix_free() const
    {
      return mf_data;
    }

    std::size_t
    memory_consumption() const
    {
      std::size_t result = sizeof(*this);
      result += mf_data->memory_consumption();
      return result;
    }

  private:
    std::shared_ptr<const PSMF::MatrixFree<dim, Number>> mf_data;
  };
} // namespace PSMF


#endif // LAPLACE_OPERATOR_CUH