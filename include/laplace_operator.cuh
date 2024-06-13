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

#include <deal.II/lac/diagonal_matrix.h>

#include "cuda_fe_evaluation.cuh"
#include "cuda_matrix_free.cuh"
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

    template <bool is_ghost>
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
        laplace_kernel_basic<dim, fe_degree, Number, kernel, is_ghost>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem));
    }

    template <typename VectorType, typename DataType, bool is_ghost>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3       &block_dim,
                cudaStream_t      stream) const
    {
      laplace_kernel_basic<dim, fe_degree, Number, kernel, is_ghost>
        <<<grid_dim, block_dim, shared_mem, stream>>>(src.get_values(),
                                                      dst.get_values(),
                                                      gpu_data);
    }
  };

  template <int dim, int fe_degree, typename Number>
  struct LocalLaplace<dim, fe_degree, Number, LaplaceVariant::BasicCell>
  {
    static constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;

    mutable std::size_t shared_mem;

    LocalLaplace()
      : shared_mem(0){};

    template <bool is_ghost>
    void
    setup_kernel(const unsigned int patch_per_block) const
    {
      constexpr unsigned int n = dim - 1;

      shared_mem = 0;

      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst
      shared_mem += 2 * patch_per_block * local_dim * sizeof(Number);
      // local_mass, local_derivative
      shared_mem +=
        2 * patch_per_block * n_dofs_1d * n_dofs_1d * 3 * sizeof(Number);
      // temp
      shared_mem += n * patch_per_block * local_dim * sizeof(Number);

      AssertCuda(
        cudaFuncSetAttribute(laplace_kernel_basic_cell<dim,
                                                       fe_degree,
                                                       Number,
                                                       LaplaceVariant::Basic>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shared_mem));
    }

    template <typename VectorType, typename DataType, bool is_ghost>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3       &block_dim,
                cudaStream_t      stream) const
    {
      laplace_kernel_basic_cell<dim, fe_degree, Number, LaplaceVariant::Basic>
        <<<grid_dim, block_dim, shared_mem, stream>>>(src.get_values(),
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

    template <bool is_ghost>
    void
    setup_kernel(const unsigned int patch_per_block) const
    {
      shared_mem = 0;

      constexpr unsigned int n_dofs_1d_padding = n_dofs_1d + Util::padding;
      constexpr unsigned int local_dim_padding =
        Util::pow(n_dofs_1d, dim - 1) * n_dofs_1d_padding;

      // local_src, local_dst
      shared_mem += 2 * patch_per_block * local_dim_padding * sizeof(Number);
      // local_mass, local_derivative
      shared_mem += 2 * patch_per_block * n_dofs_1d * n_dofs_1d_padding * 3 *
                    sizeof(Number);
      // temp
      shared_mem +=
        (dim - 1) * patch_per_block * local_dim_padding * sizeof(Number);

      AssertCuda(cudaFuncSetAttribute(
        laplace_kernel_tensorcore<dim,
                                  fe_degree,
                                  Number,
                                  LaplaceVariant::TensorCore>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem));
    }

    template <typename VectorType, typename DataType, bool is_ghost>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3       &block_dim,
                cudaStream_t      stream) const
    {
      laplace_kernel_tensorcore<dim,
                                fe_degree,
                                Number,
                                LaplaceVariant::TensorCore>
        <<<grid_dim, block_dim, shared_mem, stream>>>(src.get_values(),
                                                      dst.get_values(),
                                                      gpu_data);
    }
  };


  template <int dim, int fe_degree, typename Number>
  struct LocalLaplace<dim, fe_degree, Number, LaplaceVariant::TensorCoreMMA>
  {
    static constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;

    mutable std::size_t shared_mem;

    LocalLaplace()
      : shared_mem(0){};

    template <bool is_ghost>
    void
    setup_kernel(const unsigned int patch_per_block) const
    {
      shared_mem = 0;

      constexpr unsigned int n_dofs_1d_padding = n_dofs_1d + Util::padding;
      constexpr unsigned int local_dim_padding =
        Util::pow(n_dofs_1d, dim - 1) * n_dofs_1d_padding;

      // local_src, local_dst
      shared_mem += 2 * patch_per_block * local_dim_padding * sizeof(Number);
      // local_mass, local_derivative
      shared_mem += 2 * patch_per_block * n_dofs_1d * n_dofs_1d_padding * 3 *
                    sizeof(Number);
      // temp
      shared_mem +=
        (dim - 1) * patch_per_block * local_dim_padding * sizeof(Number);

      AssertCuda(cudaFuncSetAttribute(
        laplace_kernel_tensorcore_mma<dim,
                                      fe_degree,
                                      Number,
                                      LaplaceVariant::TensorCoreMMA>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem));
    }

    template <typename VectorType, typename DataType, bool is_ghost>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3       &block_dim,
                cudaStream_t      stream) const
    {
      laplace_kernel_tensorcore_mma<dim,
                                    fe_degree,
                                    Number,
                                    LaplaceVariant::TensorCoreMMA>
        <<<grid_dim, block_dim, shared_mem, stream>>>(src.get_values(),
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
      auto locally_owned_dofs = dof_handler->locally_owned_mg_dofs(mg_level);
      auto locally_relevant_dofs =
        DoFTools::extract_locally_relevant_level_dofs(*dof_handler, mg_level);

      vec.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    }

    void
    compute_residual(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src,
      const Function<dim, Number> &rhs_function,
      const Function<dim, Number> &exact_solution,
      const unsigned int) const
    {
      const MappingQGeneric<dim> mapping(fe_degree);
      AffineConstraints<Number>  dummy;
      dummy.close();

      typename dealii::MatrixFree<dim, Number>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
        dealii::MatrixFree<dim, Number>::AdditionalData::none;
      additional_data.mapping_update_flags =
        (update_gradients | update_JxW_values | update_quadrature_points);
      additional_data.mapping_update_flags_inner_faces =
        (update_gradients | update_JxW_values | update_normal_vectors);
      additional_data.mapping_update_flags_boundary_faces =
        (update_gradients | update_JxW_values | update_normal_vectors |
         update_quadrature_points);

      dealii::MatrixFree<dim, Number> data;
      data.reinit(mapping,
                  *dof_handler,
                  dummy,
                  QGauss<1>(fe_degree + 1),
                  additional_data);

      dst = 0.;

      const unsigned int n_dofs = src.size();

      LinearAlgebra::distributed::Vector<Number, MemorySpace::Host>
        system_rhs_host(n_dofs);
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
        system_rhs_dev(n_dofs);

      LinearAlgebra::ReadWriteVector<Number> rw_vector(n_dofs);


      dealii::FEEvaluation<dim, fe_degree> phi(data);

      for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
        {
          phi.reinit(cell);
          for (const unsigned int q : phi.quadrature_point_indices())
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

      dealii::FEFaceEvaluation<dim, fe_degree> phi_face(data, true);
      for (unsigned int face = data.n_inner_face_batches();
           face < data.n_inner_face_batches() + data.n_boundary_face_batches();
           ++face)
        {
          phi_face.reinit(face);

          const VectorizedArray<double> h_inner =
            1. / std::abs((phi_face.get_normal_vector(0) *
                           phi_face.inverse_jacobian(0))[dim - 1]);
          const auto one_over_h = (0.5 / h_inner) + (0.5 / h_inner);
          const auto gamma = fe_degree == 0 ? 1 : fe_degree * (fe_degree + 1);
          const VectorizedArray<double> sigma = 2.0 * gamma * one_over_h;

          for (const unsigned int q : phi_face.quadrature_point_indices())
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

                  if (data.get_boundary_id(face) == 0)
                    test_value[v] = 1.0 * exact_solution.value(single_point);
                  else
                    {
                      Tensor<1, dim> normal;
                      for (unsigned int d = 0; d < dim; ++d)
                        normal[d] = phi_face.get_normal_vector(q)[d][v];
                      test_normal_derivative[v] =
                        -normal * exact_solution.gradient(single_point);
                    }
                }
              phi_face.submit_value(test_value * sigma - test_normal_derivative,
                                    q);
              phi_face.submit_normal_derivative(-1.0 * test_value, q);
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
      return result;
    }

  private:
    std::shared_ptr<const LevelVertexPatch<dim, fe_degree, Number>> data;
    const DoFHandler<dim>                                          *dof_handler;
    unsigned int                                                    mg_level;
  };



  template <int dim, int fe_degree, typename Number, bool diag = false>
  class LocalLaplaceOperator
  {
  public:
    static const unsigned int n_dofs_1d    = fe_degree + 1;
    static const unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
    static const unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);

    static const unsigned int cells_per_block =
      PSMF::cells_per_block_shmem(dim, fe_degree);


    LocalLaplaceOperator()
    {}

    __device__ void
    operator()(const unsigned int                                  cell,
               const typename PSMF::MatrixFree<dim, Number>::Data *gpu_data,
               PSMF::SharedData<dim, Number>                      *shared_data,
               const Number                                       *src,
               Number                                             *dst) const
    {
      PSMF::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> fe_eval(
        cell, gpu_data, shared_data);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(false, true);
      fe_eval.submit_gradient(fe_eval.get_gradient());
      fe_eval.integrate(false, true);
      fe_eval.distribute_local_to_global(dst);
    }
  };


  template <int dim, int fe_degree, typename Number>
  class LocalLaplaceOperator<dim, fe_degree, Number, true>
  {
  public:
    static const unsigned int n_dofs_1d    = fe_degree + 1;
    static const unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
    static const unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);

    static const unsigned int cells_per_block =
      PSMF::cells_per_block_shmem(dim, fe_degree);


    LocalLaplaceOperator()
    {}

    __device__ void
    operator()(const unsigned int                                  cell,
               const typename PSMF::MatrixFree<dim, Number>::Data *gpu_data,
               PSMF::SharedData<dim, Number>                      *shared_data,
               const Number *,
               Number *dst) const
    {
      PSMF::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> fe_eval(
        cell, gpu_data, shared_data);

      Number my_diagonal = 0.0;

      const unsigned int tid = compute_index<dim, n_dofs_1d>();
      for (unsigned int i = 0; i < n_local_dofs; ++i)
        {
          fe_eval.submit_dof_value(i == tid ? 1.0 : 0.0);
          fe_eval.evaluate(false, true);
          fe_eval.submit_gradient(fe_eval.get_gradient());
          fe_eval.integrate(false, true);
          if (tid == i)
            my_diagonal = fe_eval.get_value();
        }
      fe_eval.submit_dof_value(my_diagonal);
      fe_eval.distribute_local_to_global(dst);
    }
  };



  template <int dim, int fe_degree, typename Number, bool diag = false>
  class LocalLaplaceEstimator
  {
  public:
    static const unsigned int n_dofs_1d    = fe_degree + 1;
    static const unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
    static const unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);

    static const unsigned int cells_per_block =
      PSMF::cells_per_block_shmem(dim, fe_degree);


    LocalLaplaceEstimator()
    {}

    __device__ void
    operator()(const unsigned int                                  cell,
               const typename PSMF::MatrixFree<dim, Number>::Data *gpu_data,
               PSMF::SharedData<dim, Number>                      *shared_data,
               const Number                                       *src,
               Number                                             *dst) const
    {
      PSMF::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> fe_eval(
        cell, gpu_data, shared_data);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate_hessian();

      auto trace = fe_eval.get_trace_hessian();
      auto h_k   = fe_eval.inv_jac[0];
      auto t     = h_k * trace;

      fe_eval.submit_value(t * t * 2);

      auto val = fe_eval.get_value();

      atomicAdd(&dst[cell], val);
    }
  };



  template <int dim, int fe_degree, typename Number, bool diag = false>
  class LocalLaplaceBDOperator
  {
  public:
    static const unsigned int n_dofs_1d    = fe_degree + 1;
    static const unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
    static const unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);

    static const unsigned int cells_per_block = 1;


    LocalLaplaceBDOperator()
    {}

    __device__ Number
    get_penalty_factor() const
    {
      return 1.0 * fe_degree * (fe_degree + 1);
    }

    __device__ void
    operator()(const unsigned int                                  face,
               const typename PSMF::MatrixFree<dim, Number>::Data *gpu_data,
               PSMF::SharedData<dim, Number>                      *shared_data,
               const Number                                       *src,
               Number                                             *dst) const
    {
      PSMF::FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> fe_eval(
        face, gpu_data, shared_data, true);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true, true);

      auto hi    = fabs(fe_eval.inverse_length_normal_to_face());
      auto sigma = hi * get_penalty_factor();

      auto u_inner                 = fe_eval.get_value();
      auto normal_derivative_inner = fe_eval.get_normal_derivative();
      auto test_by_value = 2 * u_inner * sigma - normal_derivative_inner;

      fe_eval.submit_value(test_by_value);
      fe_eval.submit_normal_derivative(-u_inner);

      fe_eval.integrate(true, true);
      fe_eval.distribute_local_to_global(dst);
    }
  };



  template <int dim, int fe_degree, typename Number>
  class LocalLaplaceBDOperator<dim, fe_degree, Number, true>
  {
  public:
    static const unsigned int n_dofs_1d    = fe_degree + 1;
    static const unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
    static const unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);

    static const unsigned int cells_per_block = 1;


    LocalLaplaceBDOperator()
    {}

    __device__ Number
    get_penalty_factor() const
    {
      return 1.0 * fe_degree * (fe_degree + 1);
    }

    __device__ void
    operator()(const unsigned int                                  face,
               const typename PSMF::MatrixFree<dim, Number>::Data *gpu_data,
               PSMF::SharedData<dim, Number>                      *shared_data,
               const Number                                       *src,
               Number                                             *dst) const
    {
      PSMF::FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> fe_eval(
        face, gpu_data, shared_data, true);

      Number my_diagonal = 0.0;

      const unsigned int tid = compute_index<dim, n_dofs_1d>();
      for (unsigned int i = 0; i < n_local_dofs; ++i)
        {
          fe_eval.submit_dof_value(i == tid ? 1.0 : 0.0);
          fe_eval.evaluate(true, true);

          auto hi    = fabs(fe_eval.inverse_length_normal_to_face());
          auto sigma = hi * get_penalty_factor();

          auto u_inner                 = fe_eval.get_value();
          auto normal_derivative_inner = fe_eval.get_normal_derivative();
          auto test_by_value = 2 * u_inner * sigma - normal_derivative_inner;

          fe_eval.submit_value(test_by_value);
          fe_eval.submit_normal_derivative(-u_inner);

          fe_eval.integrate(true, true);

          if (tid == i)
            my_diagonal = fe_eval.get_value();
        }
      fe_eval.submit_dof_value(my_diagonal);
      fe_eval.distribute_local_to_global(dst);
    }
  };



  template <int dim, int fe_degree, typename Number, bool diag = false>
  class LocalLaplaceFaceOperator
  {
  public:
    static const unsigned int n_dofs_1d = fe_degree + 1;
    static const unsigned int n_local_dofs =
      Utilities::pow(fe_degree + 1, dim) * 2;
    static const unsigned int n_q_points =
      Utilities::pow(fe_degree + 1, dim) * 2;

    static const unsigned int cells_per_block = 1;


    LocalLaplaceFaceOperator()
    {}

    __device__ Number
    get_penalty_factor() const
    {
      return 1.0 * fe_degree * (fe_degree + 1);
    }

    __device__ void
    operator()(const unsigned int                                  face,
               const typename PSMF::MatrixFree<dim, Number>::Data *gpu_data,
               PSMF::SharedData<dim, Number>                      *shared_data,
               const Number                                       *src,
               Number                                             *dst) const
    {
      PSMF::FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>
        phi_inner(face, gpu_data, shared_data, true);
      PSMF::FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>
        phi_outer(face, gpu_data, shared_data, false);

      phi_inner.read_dof_values(src);
      phi_inner.evaluate(true, true);

      phi_outer.read_dof_values(src);
      phi_outer.evaluate(true, true);

      auto hi    = 0.5 * (fabs(phi_inner.inverse_length_normal_to_face()) +
                       fabs(phi_outer.inverse_length_normal_to_face()));
      auto sigma = hi * get_penalty_factor();

      auto solution_jump = phi_inner.get_value() - phi_outer.get_value();
      auto average_normal_derivative =
        0.5 *
        (phi_inner.get_normal_derivative() + phi_outer.get_normal_derivative());
      auto test_by_value = solution_jump * sigma - average_normal_derivative;


      phi_inner.submit_value(test_by_value);
      phi_outer.submit_value(-test_by_value);

      phi_inner.submit_normal_derivative(-solution_jump * 0.5);
      phi_outer.submit_normal_derivative(-solution_jump * 0.5);

      phi_inner.integrate(true, true);
      phi_inner.distribute_local_to_global(dst);

      phi_outer.integrate(true, true);
      phi_outer.distribute_local_to_global(dst);
    }
  };



  template <int dim, int fe_degree, typename Number>
  class LocalLaplaceFaceOperator<dim, fe_degree, Number, true>
  {
  public:
    static const unsigned int n_dofs_1d = fe_degree + 1;
    static const unsigned int n_local_dofs =
      Utilities::pow(fe_degree + 1, dim) * 2;
    static const unsigned int n_q_points =
      Utilities::pow(fe_degree + 1, dim) * 2;

    static const unsigned int cells_per_block = 1;


    LocalLaplaceFaceOperator()
    {}

    __device__ Number
    get_penalty_factor() const
    {
      return 1.0 * fe_degree * (fe_degree + 1);
    }

    __device__ void
    operator()(const unsigned int                                  face,
               const typename PSMF::MatrixFree<dim, Number>::Data *gpu_data,
               PSMF::SharedData<dim, Number>                      *shared_data,
               const Number                                       *src,
               Number                                             *dst) const
    {
      PSMF::FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>
        phi_inner(face, gpu_data, shared_data, true);
      PSMF::FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>
        phi_outer(face, gpu_data, shared_data, false);

      auto hi    = 0.5 * (fabs(phi_inner.inverse_length_normal_to_face()) +
                       fabs(phi_outer.inverse_length_normal_to_face()));
      auto sigma = hi * get_penalty_factor();

      Number my_diagonal = 0.0;

      const unsigned int tid = compute_index<dim, n_dofs_1d>();
      for (unsigned int i = 0; i < n_local_dofs / 2; ++i)
        {
          phi_inner.submit_dof_value(i == tid ? 1.0 : 0.0);
          phi_inner.evaluate(true, true);

          auto solution_jump = phi_inner.get_value();
          auto average_normal_derivative =
            0.5 * phi_inner.get_normal_derivative();
          auto test_by_value =
            solution_jump * sigma - average_normal_derivative;

          phi_inner.submit_value(test_by_value);
          phi_inner.submit_normal_derivative(-solution_jump * 0.5);

          phi_inner.integrate(true, true);

          if (tid == i)
            my_diagonal = phi_inner.get_value();
        }

      phi_inner.submit_dof_value(my_diagonal);
      phi_inner.distribute_local_to_global(dst);

      for (unsigned int i = 0; i < n_local_dofs / 2; ++i)
        {
          phi_outer.submit_dof_value(i == tid ? 1.0 : 0.0);
          phi_outer.evaluate(true, true);

          auto solution_jump = -phi_outer.get_value();
          auto average_normal_derivative =
            0.5 * phi_outer.get_normal_derivative();
          auto test_by_value =
            solution_jump * sigma - average_normal_derivative;

          phi_outer.submit_value(-test_by_value);
          phi_outer.submit_normal_derivative(-solution_jump * 0.5);

          phi_outer.integrate(true, true);

          if (tid == i)
            my_diagonal = phi_outer.get_value();
        }

      phi_outer.submit_dof_value(my_diagonal);
      phi_outer.distribute_local_to_global(dst);
    }
  };


  template <int dim, int fe_degree, typename Number>
  class LaplaceDGOperator : public Subscriptor
  {
  public:
    using value_type = Number;

    LaplaceDGOperator()
    {
      inverse_diagonal_matrix = std::make_shared<DiagonalMatrix<
        LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>>>();
    }

    void
    initialize(std::shared_ptr<const MatrixFree<dim, Number>> data_,
               const DoFHandler<dim>                         &mg_dof,
               const unsigned int level = numbers::invalid_unsigned_int)
    {
      data        = data_;
      dof_handler = &mg_dof;
      mg_level    = level;

      if (mg_level == numbers::invalid_unsigned_int)
        n_dofs = dof_handler->n_dofs();
      else
        n_dofs = dof_handler->n_dofs(mg_level);
    }

    void
    vmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
          const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
            &src) const
    {
      dst = 0.;
      LocalLaplaceOperator<dim, fe_degree, Number> laplace_operator;
      data->cell_loop(laplace_operator, src, dst);

      LocalLaplaceBDOperator<dim, fe_degree, Number> laplace_bd_operator;
      data->boundary_face_loop(laplace_bd_operator, src, dst);

      LocalLaplaceFaceOperator<dim, fe_degree, Number> laplace_face_operator;
      data->inner_face_loop(laplace_face_operator, src, dst);
    }

    void
    estimate(LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
             const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
               &src) const
    {
      dst = 0.;
      LocalLaplaceEstimator<dim, fe_degree, Number> laplace_operator;
      data->cell_loop(laplace_operator, src, dst);
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
      if (mg_level == numbers::invalid_unsigned_int)
        {
          auto locally_owned_dofs = dof_handler->locally_owned_dofs();
          auto locally_relevant_dofs =
            DoFTools::extract_locally_relevant_dofs(*dof_handler);
          vec.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
        }
      else
        {
          auto locally_owned_dofs =
            dof_handler->locally_owned_mg_dofs(mg_level);
          auto locally_relevant_dofs =
            DoFTools::extract_locally_relevant_level_dofs(*dof_handler,
                                                          mg_level);
          vec.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
        }
    }

    unsigned int
    m() const
    {
      return n_dofs;
    }

    // we cannot access matrix elements of a matrix free operator directly.
    Number
    el(const unsigned int, const unsigned int) const
    {
      ExcNotImplemented();
      return -1000000000000000000;
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

    void
    compute_residual(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src,
      const Function<dim, Number> &rhs_function,
      const Function<dim, Number> &exact_solution,
      const unsigned int) const
    {
      const MappingQGeneric<dim> mapping(fe_degree);
      AffineConstraints<Number>  dummy;
      dummy.close();

      typename dealii::MatrixFree<dim, Number>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
        dealii::MatrixFree<dim, Number>::AdditionalData::none;
      additional_data.mapping_update_flags =
        (update_gradients | update_JxW_values | update_quadrature_points);
      additional_data.mapping_update_flags_inner_faces =
        (update_gradients | update_JxW_values | update_normal_vectors);
      additional_data.mapping_update_flags_boundary_faces =
        (update_gradients | update_JxW_values | update_normal_vectors |
         update_quadrature_points);

      dealii::MatrixFree<dim, Number> data;
      data.reinit(mapping,
                  *dof_handler,
                  dummy,
                  QGauss<1>(fe_degree + 1),
                  additional_data);

      dst = 0.;

      const unsigned int n_dofs = src.size();

      LinearAlgebra::distributed::Vector<Number, MemorySpace::Host>
        system_rhs_host(n_dofs);
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
        system_rhs_dev(n_dofs);

      LinearAlgebra::ReadWriteVector<Number> rw_vector(n_dofs);


      dealii::FEEvaluation<dim, fe_degree> phi(data);

      for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
        {
          phi.reinit(cell);
          for (const unsigned int q : phi.quadrature_point_indices())
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

      dealii::FEFaceEvaluation<dim, fe_degree> phi_face(data, true);
      for (unsigned int face = data.n_inner_face_batches();
           face < data.n_inner_face_batches() + data.n_boundary_face_batches();
           ++face)
        {
          phi_face.reinit(face);

          const VectorizedArray<double> h_inner =
            1. / std::abs((phi_face.get_normal_vector(0) *
                           phi_face.inverse_jacobian(0))[dim - 1]);
          const auto one_over_h = (0.5 / h_inner) + (0.5 / h_inner);
          const auto gamma = fe_degree == 0 ? 1 : fe_degree * (fe_degree + 1);
          const VectorizedArray<double> sigma = 2.0 * gamma * one_over_h;

          for (const unsigned int q : phi_face.quadrature_point_indices())
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

                  if (data.get_boundary_id(face) == 0)
                    test_value[v] = 1.0 * exact_solution.value(single_point);
                  else
                    {
                      Tensor<1, dim> normal;
                      for (unsigned int d = 0; d < dim; ++d)
                        normal[d] = phi_face.get_normal_vector(q)[d][v];
                      test_normal_derivative[v] =
                        -normal * exact_solution.gradient(single_point);
                    }
                }
              phi_face.submit_value(test_value * sigma - test_normal_derivative,
                                    q);
              phi_face.submit_normal_derivative(-1.0 * test_value, q);
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


    void
    compute_diagonal()
    {
      auto &inv_diag = inverse_diagonal_matrix->get_vector();

      inv_diag.reinit(n_dofs);

      LocalLaplaceOperator<dim, fe_degree, Number, true> laplace_operator;
      data->cell_loop(laplace_operator, inv_diag, inv_diag);

      LocalLaplaceBDOperator<dim, fe_degree, Number, true> laplace_bd_operator;
      data->boundary_face_loop(laplace_bd_operator, inv_diag, inv_diag);

      LocalLaplaceFaceOperator<dim, fe_degree, Number, true>
        laplace_face_operator;
      data->inner_face_loop(laplace_face_operator, inv_diag, inv_diag);

      vector_invert(inv_diag);

      diagonal_is_available = true;
    }

    const std::shared_ptr<DiagonalMatrix<
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>>>
    get_diagonal_inverse() const
    {
      Assert(diagonal_is_available == true, ExcNotInitialized());
      return inverse_diagonal_matrix;
    }


  private:
    std::shared_ptr<const MatrixFree<dim, Number>> data;
    const DoFHandler<dim>                         *dof_handler;
    unsigned int                                   mg_level;
    unsigned int                                   n_dofs;

    std::shared_ptr<DiagonalMatrix<
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>>>
         inverse_diagonal_matrix;
    bool diagonal_is_available;
  };


  template <int dim, int fe_degree, typename Number>
  class LaplaceDGEdgeOperator : public Subscriptor
  {
  public:
    using value_type = Number;

    LaplaceDGEdgeOperator()
    {}

    void
    initialize(std::shared_ptr<const MatrixFree<dim, Number>> data_,
               const DoFHandler<dim>                         &mg_dof,
               const unsigned int                             level)
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

      LocalLaplaceFaceOperator<dim, fe_degree, Number> laplace_face_operator;
      data->inner_face_loop(laplace_face_operator, src, dst);
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
      auto locally_owned_dofs = dof_handler->locally_owned_mg_dofs(mg_level);
      auto locally_relevant_dofs =
        DoFTools::extract_locally_relevant_level_dofs(*dof_handler, mg_level);

      vec.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
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

  private:
    std::shared_ptr<const MatrixFree<dim, Number>> data;
    const DoFHandler<dim>                         *dof_handler;
    unsigned int                                   mg_level;
  };
} // namespace PSMF


#endif // LAPLACE_OPERATOR_CUH
