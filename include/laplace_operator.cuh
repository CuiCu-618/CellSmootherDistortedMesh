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

#include <deal.II/fe/fe_interface_values.h>

#include "patch_base.cuh"

using namespace dealii;

namespace PSMF
{

  template <int dim, int fe_degree, typename Number, LaplaceVariant kernel>
  struct LocalLaplace
  {
#ifdef TENSORCORE
    static constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;
#else
    static constexpr unsigned int n_dofs_1d = 2 * fe_degree + 1;
#endif

    mutable std::size_t shared_mem;

    LocalLaplace()
      : shared_mem(0){};

    void
    setup_kernel(const unsigned int patch_per_block) const
    {
      constexpr unsigned int n =
        kernel == LaplaceVariant::Basic ? (dim - 1) : 2;

      shared_mem = 0;

      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst
      shared_mem += 2 * patch_per_block * local_dim * sizeof(Number);
      // local_mass, local_derivative, local_bilaplace
      shared_mem +=
        3 * patch_per_block * n_dofs_1d * n_dofs_1d * dim * sizeof(Number);
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

      data->copy_constrained_values(src, dst);
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
      // src.update_ghost_values();

      const unsigned int n_dofs = src.size();

      LinearAlgebra::distributed::Vector<Number, MemorySpace::Host>
        system_rhs_host(n_dofs);
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
        system_rhs_dev(n_dofs);

      LinearAlgebra::ReadWriteVector<Number> rw_vector(n_dofs);

      AffineConstraints<Number> constraints;
      constraints.clear();
      VectorTools::interpolate_boundary_values(*dof_handler,
                                               0,
                                               exact_solution,
                                               constraints);
      constraints.close();

      const QGauss<dim>      quadrature_formula(fe_degree + 1);
      FEValues<dim>          fe_values(dof_handler->get_fe(),
                              quadrature_formula,
                              update_values | update_quadrature_points |
                                update_JxW_values);
      FEInterfaceValues<dim> fe_interface_values(
        dof_handler->get_fe(),
        QGauss<dim - 1>(fe_degree + 1),
        update_values | update_gradients | update_quadrature_points |
          update_hessians | update_JxW_values | update_normal_vectors);

      const unsigned int dofs_per_cell =
        dof_handler->get_fe().n_dofs_per_cell();

      const unsigned int        n_q_points = quadrature_formula.size();
      Vector<Number>            cell_rhs(dofs_per_cell);
      std::vector<unsigned int> local_dof_indices(dofs_per_cell);
      std::vector<Number>       rhs_values(n_q_points);

      auto begin = dof_handler->begin_mg(mg_level);
      auto end   = dof_handler->end_mg(mg_level);

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

      for (auto cell = begin; cell != end; ++cell)
        if (cell->is_locally_owned_on_level())
          {
            for (const unsigned int face_no : cell->face_indices())
              if (cell->at_boundary(face_no))
                {
                  fe_interface_values.reinit(cell, face_no);

                  const unsigned int n_interface_dofs =
                    fe_interface_values.n_current_interface_dofs();
                  Vector<Number> cell_rhs_face(n_interface_dofs);
                  cell_rhs_face = 0;

                  const auto &q_points =
                    fe_interface_values.get_quadrature_points();
                  const std::vector<double> &JxW =
                    fe_interface_values.get_JxW_values();
                  const std::vector<Tensor<1, dim>> &normals =
                    fe_interface_values.get_normal_vectors();

                  std::vector<Tensor<1, dim>> exact_gradients(q_points.size());
                  exact_solution.gradient_list(q_points, exact_gradients);

                  const unsigned int p = fe_degree;
                  const auto         h = cell->extent_in_direction(
                    GeometryInfo<dim>::unit_normal_direction[face_no]);
                  const auto   one_over_h   = (0.5 / h) + (0.5 / h);
                  const auto   gamma        = p == 0 ? 1 : p * (p + 1);
                  const double gamma_over_h = 2.0 * gamma * one_over_h;

                  for (unsigned int qpoint = 0; qpoint < q_points.size();
                       ++qpoint)
                    {
                      const auto &n = normals[qpoint];

                      for (unsigned int i = 0; i < n_interface_dofs; ++i)
                        {
                          const double av_hessian_i_dot_n_dot_n =
                            (fe_interface_values.average_of_shape_hessians(
                               i, qpoint) *
                             n * n);
                          const double jump_grad_i_dot_n =
                            (fe_interface_values.jump_in_shape_gradients(
                               i, qpoint) *
                             n);
                          cell_rhs_face(i) +=
                            (-av_hessian_i_dot_n_dot_n * // - {grad^2 v n n }
                               (exact_gradients[qpoint] *
                                n)                 //   (grad u_exact . n)
                             +                     // +
                             gamma_over_h          //  gamma/h
                               * jump_grad_i_dot_n // [grad v n]
                               * (exact_gradients[qpoint] *
                                  n) // (grad u_exact . n)
                             ) *
                            JxW[qpoint]; // dx
                        }
                    }

                  auto dof_indices =
                    fe_interface_values.get_interface_dof_indices();
                  constraints.distribute_local_to_global(cell_rhs_face,
                                                         dof_indices,
                                                         system_rhs_host);
                }
          }

      system_rhs_host.compress(VectorOperation::add);
      rw_vector.import(system_rhs_host, VectorOperation::insert);
      system_rhs_dev.import(rw_vector, VectorOperation::insert);

      // system_rhs_host = 0.;
      // system_rhs_host[10] = 1.;
      // rw_vector.import(system_rhs_host, VectorOperation::insert);
      // system_rhs_dev.import(rw_vector, VectorOperation::insert);

      // vmult(dst, system_rhs_dev);
      // dst.print(std::cout);

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
      return result;
    }

  private:
    std::shared_ptr<const LevelVertexPatch<dim, fe_degree, Number>> data;
    const DoFHandler<dim>                                          *dof_handler;
    unsigned int                                                    mg_level;
  };
} // namespace PSMF


#endif // LAPLACE_OPERATOR_CUH
