/**
 * @file utilities.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief collection of solvers
 * @version 1.0
 * @date 2022-12-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef SOLVER_CUH
#define SOLVER_CUH

#include <deal.II/base/function.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/multigrid.h>

#include <functional>

#include "cell_base.cuh"
#include "cell_smoother.cuh"
#include "cuda_matrix_free.cuh"
#include "cuda_mg_transfer.cuh"
#include "laplace_operator.cuh"
// #include "patch_base.cuh"
// #include "patch_smoother.cuh"

using namespace dealii;

namespace PSMF
{
  template <bool is_zero, typename Number>
  __global__ void
  set_inhomogeneous_dofs(const unsigned int *indicex,
                         const Number       *values,
                         const unsigned int  n_inhomogeneous_dofs,
                         Number             *dst)
  {
    const unsigned int dof =
      threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);

    if (dof < n_inhomogeneous_dofs)
      {
        if (is_zero)
          dst[indicex[dof]] = 0;
        else
          dst[indicex[dof]] = values[dof];
      }
  }

  template <bool is_d2f, typename number, typename number2>
  __global__ void
  copy_vector(number *dst, const number2 *src, const unsigned n_dofs)
  {
    const unsigned int dof =
      threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);

    if (dof < n_dofs)
      {
        if (is_d2f)
          dst[dof] = __double2float_rn(src[dof]);
        else
          dst[dof] = src[dof];
      }
  }

  // A coarse solver defined via the smoother
  template <typename VectorType, typename SmootherType>
  class MGCoarseFromSmoother : public MGCoarseGridBase<VectorType>
  {
  public:
    MGCoarseFromSmoother(const SmootherType &mg_smoother, const bool is_empty)
      : smoother(mg_smoother)
      , is_empty(is_empty)
    {}

    virtual void
    operator()(const unsigned int level,
               VectorType        &dst,
               const VectorType  &src) const
    {
      if (is_empty)
        return;
      smoother[level].vmult(dst, src);
    }

    const SmootherType &smoother;
    const bool          is_empty;
  };


  // coarse solver
  template <typename MatrixType, typename VectorType>
  class MGCoarseIterative : public MGCoarseGridBase<VectorType>
  {
  public:
    MGCoarseIterative()
    {}

    void
    initialize(const MatrixType &matrix)
    {
      coarse_matrix = &matrix;
    }

    virtual void
    operator()(const unsigned int level,
               VectorType        &dst,
               const VectorType  &src) const
    {
      ReductionControl     solver_control(1000, 1e-15, 1e-10);
      SolverCG<VectorType> solver_coarse(solver_control);
      solver_coarse.solve(*coarse_matrix, dst, src, PreconditionIdentity());
    }

    const MatrixType *coarse_matrix;
  };

  struct SolverData
  {
    std::string solver_name = "";

    int    n_iteration      = 0;
    double n_step           = 0;
    double residual         = 0.;
    double reduction_rate   = 0.;
    double convergence_rate = 0.;
    double l2_error         = 0.;
    int    mem_usage        = 0.;
    double timing           = 0.;
    double perf             = 0.;

    double cg_it    = 0;
    double cg_error = 0;

    std::string
    print_comp()
    {
      std::ostringstream oss;

      oss.width(12);
      oss.precision(4);
      oss.setf(std::ios::left);
      oss.setf(std::ios::scientific);

      oss << std::left << std::setw(12) << solver_name << std::setprecision(4)
          << std::setw(12) << timing << std::setprecision(4) << std::setw(12)
          << perf << std::endl;

      return oss.str();
    }

    std::string
    print_solver()
    {
      std::ostringstream oss;

      oss.width(12);
      oss.precision(4);
      oss.setf(std::ios::left);

      oss << std::left << std::setw(12) << solver_name << std::setw(4)
          << n_iteration << std::setw(8) << n_step;

      oss.setf(std::ios::scientific);

      // if (CT::SETS_ == "error_analysis")
      oss << std::setprecision(4) << std::setw(12) << timing << std::left
          << std::setprecision(4) << std::setw(12) << residual
          << std::setprecision(4) << std::setw(12) << reduction_rate
          << std::setprecision(4) << std::setw(12) << convergence_rate;

      oss << std::left << std::setw(8) << mem_usage << std::endl;

      return oss.str();
    }
  };

  template <int       dim,
            int       fe_degree,
            DoFLayout dof_layout,
            typename Number,
            LaplaceVariant  lapalace_kernel,
            LaplaceVariant  smooth_vmult,
            SmootherVariant smooth_inverse,
            typename Number2>
  class MultigridSolver
  {
  public:
    using VectorType =
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>;
    using VectorTypeSP =
      LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA>;
    using MatrixType   = LaplaceDGOperator<dim, fe_degree, Number>;
    using MatrixTypeSP = LaplaceDGOperator<dim, fe_degree, Number2>;
    using CellSmootherType =
      CellSmoother<MatrixTypeSP, dim, fe_degree, smooth_vmult, smooth_inverse>;
    using SmootherTypeCheb = PreconditionChebyshev<MatrixTypeSP, VectorTypeSP>;
    using MatrixFreeDP     = MatrixFree<dim, Number>;
    using MatrixFreeSP     = MatrixFree<dim, Number2>;
    using CellPatchType    = LevelCellPatch<dim, fe_degree, Number2>;

    // using SmootherType =
    //   PatchSmoother<MatrixType, dim, fe_degree, smooth_vmult,
    //   smooth_inverse>;
    // using VertexPatchType  = LevelVertexPatch<dim, fe_degree, Number>;

    MultigridSolver(
      const DoFHandler<dim>                               &dof_handler,
      const MGLevelObject<std::shared_ptr<MatrixFreeDP>>  &level_mfdata,
      const MGLevelObject<std::shared_ptr<MatrixFreeSP>>  &level_mfdata_sp,
      const MGLevelObject<std::shared_ptr<CellPatchType>> &cell_data,
      const MGTransferCUDA<dim, Number2>                  &transfer,
      const Function<dim, Number>                         &boundary_values,
      const Function<dim, Number> &,
      std::shared_ptr<ConditionalOStream> pcout,
      const unsigned int                  n_cycles = 1)
      : dof_handler(&dof_handler)
      , transfer(&transfer)
      , minlevel(1)
      , maxlevel(dof_handler.get_triangulation().n_global_levels() - 1)
      , n_cycles(n_cycles)
      , analytic_solution(boundary_values)
      , pcout(pcout)
    {
      AssertDimension(fe_degree, dof_handler.get_fe().degree);

      if (smooth_inverse == PSMF::SmootherVariant::MCS ||
          smooth_inverse == PSMF::SmootherVariant::MCS_CG ||
          smooth_inverse == PSMF::SmootherVariant::MCS_PCG ||
          smooth_inverse == PSMF::SmootherVariant::Chebyshev)
        minlevel = 0;

      matrix.resize(minlevel, maxlevel);
      matrix_sp.resize(minlevel, maxlevel);

      for (unsigned int level = minlevel; level <= maxlevel; ++level)
        {
          matrix[level].initialize(level_mfdata[level], dof_handler, level);
          matrix_sp[level].initialize(level_mfdata_sp[level],
                                      dof_handler,
                                      level);

          if (level == maxlevel)
            {
              matrix[level].initialize_dof_vector(solution);
              rhs = solution;

              matrix_sp[level].initialize_dof_vector(solution_tmp);
              rhs_tmp = solution_tmp;
              rhs_tmp = 1.;
            }
        }


      {
        Timer time;
        // evaluate the right hand side in the equation, including the
        // residual from the inhomogeneous boundary conditions
        rhs = 0.;
        if (CT::SETS_ == "error_analysis")
          matrix[maxlevel].compute_rhs(rhs, solution);
        else
          rhs = 1.;

        *pcout << "RHS setup time:         " << time.wall_time() << "s"
               << std::endl;
      }

      if constexpr (smooth_inverse == PSMF::SmootherVariant::Chebyshev)
        {
          MGLevelObject<typename SmootherTypeCheb::AdditionalData>
            smoother_data;
          smoother_data.resize(minlevel, maxlevel);
          for (unsigned int level = minlevel; level <= maxlevel; ++level)
            {
              matrix_sp[level].compute_diagonal();

              smoother_data[level].smoothing_range     = 20.;
              smoother_data[level].degree              = 5;
              smoother_data[level].eig_cg_n_iterations = 20;
              smoother_data[level].preconditioner =
                matrix_sp[level].get_diagonal_inverse();
            }

          mg_smoother_cheb.initialize(matrix_sp, smoother_data);
          mg_coarse_cheb.initialize(mg_smoother_cheb);

          mg_matrix.initialize(matrix_sp);
          mg = std::make_unique<Multigrid<VectorTypeSP>>(mg_matrix,
                                                         mg_coarse_cheb,
                                                         transfer,
                                                         mg_smoother_cheb,
                                                         mg_smoother_cheb,
                                                         minlevel,
                                                         maxlevel);

          preconditioner_mg = std::make_unique<
            PreconditionMG<dim, VectorTypeSP, MGTransferCUDA<dim, Number2>>>(
            dof_handler, *mg, transfer);
        }
      else if (smooth_inverse == PSMF::SmootherVariant::MCS ||
               smooth_inverse == PSMF::SmootherVariant::MCS_CG ||
               smooth_inverse == PSMF::SmootherVariant::MCS_PCG)
        {
          MGLevelObject<typename CellSmootherType::AdditionalData>
            smoother_data;
          smoother_data.resize(minlevel, maxlevel);
          for (unsigned int level = minlevel; level <= maxlevel; ++level)
            {
              smoother_data[level].data = cell_data[level];
            }

          mg_cell_smoother.initialize(matrix_sp, smoother_data);
          mg_cell_coarse.initialize(mg_cell_smoother);

          mg_matrix.initialize(matrix_sp);
          mg = std::make_unique<Multigrid<VectorTypeSP>>(mg_matrix,
                                                         mg_cell_coarse,
                                                         transfer,
                                                         mg_cell_smoother,
                                                         mg_cell_smoother,
                                                         minlevel,
                                                         maxlevel);

          preconditioner_mg = std::make_unique<
            PreconditionMG<dim, VectorTypeSP, MGTransferCUDA<dim, Number2>>>(
            dof_handler, *mg, transfer);
        }
      else
        {
          AssertThrow(false, dealii::ExcMessage("Not implemented.\n"));
          // MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
          // smoother_data.resize(minlevel, maxlevel);
          // for (unsigned int level = minlevel; level <= maxlevel; ++level)
          //   {
          //     smoother_data[level].data = patch_data_dp[level];
          //   }
          //
          // mg_smoother.initialize(matrix, smoother_data);
          // mg_coarse.initialize(mg_smoother);
          //
          // mg_matrix.initialize(matrix);
          // mg = std::make_unique<Multigrid<VectorType>>(mg_matrix,
          //                                              mg_coarse,
          //                                              transfer_dp,
          //                                              mg_smoother,
          //                                              mg_smoother,
          //                                              minlevel,
          //                                              maxlevel);
          //
          // preconditioner_mg = std::make_unique<
          //   PreconditionMG<dim, VectorType, MGTransferCUDA<dim, Number>>>(
          //   dof_handler, *mg, transfer_dp);
        }
    }

    std::vector<SolverData>
    static_comp()
    {
      *pcout << "Testing...\n";

      std::vector<SolverData> comp_data;

      std::string comp_name = "";

      const unsigned int n_dofs = dof_handler->n_dofs();
      const unsigned int n_mv   = n_dofs < 10000000 ? 100 : 20;

      auto tester = [&](auto kernel) {
        Timer              time;
        const unsigned int N         = 5;
        double             best_time = 1e10;
        for (unsigned int i = 0; i < N; ++i)
          {
            time.restart();
            for (unsigned int i = 0; i < n_mv; ++i)
              kernel(this);
            best_time = std::min(time.wall_time() / n_mv, best_time);
          }

        SolverData data;
        data.solver_name = comp_name;
        data.timing      = best_time;
        data.perf        = n_dofs / best_time;
        comp_data.push_back(data);
      };

      for (unsigned int s = 0; s < 2; ++s)
        {
          switch (s)
            {
              case 0:
                {
                  auto kernel = std::mem_fn(&MultigridSolver::do_matvec);
                  comp_name   = "Mat-vec";
                  tester(kernel);
                  break;
                }
              case 1:
                {
                  auto kernel = std::mem_fn(&MultigridSolver::do_smooth);
                  comp_name   = "Smooth";
                  tester(kernel);
                  break;
                }
              default:
                AssertThrow(false, ExcMessage("Invalid Solver Variant."));
            }
        }

      return comp_data;
    }

    // Return the solution vector for further processing
    const VectorType &
    get_solution()
    {
      return solution;
    }

    // Implement the vmult() function needed by the preconditioner interface
    void
    vmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
          const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
            &src) const
    {
      preconditioner_mg->vmult(dst, src);
    }

    // Solve with the conjugate gradient method preconditioned by the V-cycle
    // (invoking this->vmult) and return the number of iterations and the
    // reduction rate per GMRES iteration
    std::vector<SolverData>
    solve()
    {
      *pcout << "Solving...\n";

      std::string solver_name = "GMRES";

      ReductionControl solver_control(CT::MAX_STEPS_, 1e-15, CT::REDUCE_);
      solver_control.enable_history_data();
      solver_control.log_history(true);

      SolverGMRES<VectorType> solver(solver_control);

      Timer              time;
      const unsigned int N         = 5;
      double             best_time = 1e10;

      bool is_converged = true;
      try
        {
          for (unsigned int i = 0; i < N; ++i)
            {
              time.reset();
              time.start();

              solution = 0;
              solver.solve(matrix[maxlevel], solution, rhs, *this);

              best_time = std::min(time.wall_time(), best_time);
            }
        }
      catch (...)
        {
          best_time = std::min(time.wall_time(), best_time);

          is_converged = false;

          *pcout << "\n!!! Solver not Converged within " << CT::MAX_STEPS_
                 << " steps. !!!\n\n";
        }

      auto n_iter     = solver_control.last_step();
      auto residual_0 = solver_control.initial_value();
      auto residual_n = solver_control.last_value();
      auto reduction  = solver_control.reduction();

      // *** average reduction: r_n = rho^n * r_0
      const double rho =
        std::pow(residual_n / residual_0, static_cast<double>(1. / n_iter));
      const double convergence_rate =
        1. / n_iter * std::log10(residual_0 / residual_n);

      const auto n_step = -10 * std::log10(rho);
      const auto n_frac = std::log(reduction) / std::log(rho);

      size_t free_mem, total_mem;
      AssertCuda(cudaMemGetInfo(&free_mem, &total_mem));

      int mem_usage = (total_mem - free_mem) / 1024 / 1024;

      SolverData data;
      data.solver_name      = solver_name;
      data.n_iteration      = n_iter;
      data.n_step           = n_frac;
      data.residual         = residual_n;
      data.reduction_rate   = rho;
      data.convergence_rate = convergence_rate;
      data.timing           = best_time;
      data.mem_usage        = mem_usage;

      if (smooth_inverse == PSMF::SmootherVariant::MCS_CG ||
          smooth_inverse == PSMF::SmootherVariant::MCS_PCG)
        {
          auto vec = mg_cell_smoother.smoothers[maxlevel].get_cg_solver_info();

          data.cg_it    = vec[0];
          data.cg_error = vec[1];
        }

      solver_data.push_back(data);

      if (is_converged)
        {
          auto history_data = solver_control.get_history_data();
          for (auto i = 1U; i < n_iter + 1; ++i)
            *pcout << "step " << i << ": " << history_data[i] / residual_0
                   << "\n";
        }

      return solver_data;
    }

    // run smooth in double precision
    void
    do_smooth()
    {
      if constexpr (smooth_inverse == PSMF::SmootherVariant::Chebyshev)
        (mg_smoother_cheb).smooth(maxlevel, solution_tmp, rhs_tmp);
      else if (smooth_inverse == PSMF::SmootherVariant::MCS ||
               smooth_inverse == PSMF::SmootherVariant::MCS_CG ||
               smooth_inverse == PSMF::SmootherVariant::MCS_PCG)
        (mg_cell_smoother).smooth(maxlevel, solution_tmp, rhs_tmp);
      else
        AssertThrow(false, dealii::ExcMessage("Not implemented.\n"));
      cudaDeviceSynchronize();

      // solution.print(std::cout);
      // std::cout << solution.l2_norm() << std::endl;
      //
      // AssertThrow(false, dealii::ExcMessage("debug."));
    }

    // run matrix-vector product in double precision
    void
    do_matvec()
    {
      matrix_sp[maxlevel].vmult(solution_tmp, rhs_tmp);
      cudaDeviceSynchronize();

      // AssertThrow(false, dealii::ExcMessage("debug."));
      // std::cout << rhs.l2_norm() << " " << solution.l2_norm() << std::endl;

      // for (unsigned int i = 0; i < rhs.size(); ++i)
      //   {
      //     LinearAlgebra::ReadWriteVector<double> rw_vector(rhs.size());
      //
      //     // for (unsigned int i = 0; i < rhs.size(); ++i)
      //     rw_vector[i] = 1. + 0;
      //
      //     rhs.import(rw_vector, VectorOperation::insert);
      //
      //     matrix[maxlevel].vmult(solution, rhs);
      //
      //     solution.print(std::cout);
      //     std::cout << solution.l2_norm() << std::endl;
      //
      //     // if (i == 0)
      //     break;
      //   }
      //
      // AssertThrow(false, dealii::ExcMessage("debug."));
    }

  private:
    const SmartPointer<const DoFHandler<dim>>              dof_handler;
    const SmartPointer<const MGTransferCUDA<dim, Number2>> transfer;

    MGLevelObject<MatrixType>   matrix;
    MGLevelObject<MatrixTypeSP> matrix_sp;

    /**
     * Lowest level of cells.
     */
    unsigned int minlevel;

    /**
     * Highest level of cells.
     */
    unsigned int maxlevel;

    /**
     * The solution vector
     */
    mutable VectorType   solution;
    mutable VectorTypeSP solution_tmp;

    /**
     * Original right hand side vector
     */
    mutable VectorType   rhs;
    mutable VectorTypeSP rhs_tmp;

    // MGSmootherPrecondition<MatrixType, SmootherType, VectorType> mg_smoother;

    mutable MGSmootherPrecondition<MatrixTypeSP, SmootherTypeCheb, VectorTypeSP>
      mg_smoother_cheb;

    MGSmootherPrecondition<MatrixTypeSP, CellSmootherType, VectorTypeSP>
      mg_cell_smoother;

    /**
     * The coarse solver
     */
    // MGCoarseGridApplySmoother<VectorType> mg_coarse;

    mutable MGCoarseGridApplySmoother<VectorTypeSP> mg_coarse_cheb;

    mutable MGCoarseGridApplySmoother<VectorTypeSP> mg_cell_coarse;

    mutable std::unique_ptr<
      PreconditionMG<dim, VectorTypeSP, MGTransferCUDA<dim, Number2>>>
      preconditioner_mg;

    mutable mg::Matrix<VectorTypeSP> mg_matrix;

    mutable std::unique_ptr<Multigrid<VectorTypeSP>> mg;

    /**
     * Number of cycles to be done in the FMG cycle
     */
    const unsigned int n_cycles;

    /**
     * Collection of compute times on various levels
     */
    mutable std::vector<SolverData> solver_data;

    /**
     * Function for boundary values that we keep as analytic
     * solution
     */
    const Function<dim, Number> &analytic_solution;

    std::shared_ptr<ConditionalOStream> pcout;
  };

  template <int       dim,
            int       fe_degree,
            DoFLayout dof_layout,
            typename Number,
            LaplaceVariant  lapalace_kernel,
            LaplaceVariant  smooth_vmult,
            SmootherVariant smooth_inverse>
  class MultigridSolver<dim,
                        fe_degree,
                        dof_layout,
                        Number,
                        lapalace_kernel,
                        smooth_vmult,
                        smooth_inverse,
                        Number>
  {
  public:
    using VectorType =
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>;
    using MatrixType = LaplaceDGOperator<dim, fe_degree, Number>;
    using CellSmootherType =
      CellSmoother<MatrixType, dim, fe_degree, smooth_vmult, smooth_inverse>;
    using SmootherTypeCheb = PreconditionChebyshev<MatrixType, VectorType>;
    using MatrixFree       = MatrixFree<dim, Number>;
    using CellPatchType    = LevelCellPatch<dim, fe_degree, Number>;

    // using SmootherType =
    //   PatchSmoother<MatrixType, dim, fe_degree, smooth_vmult,
    //   smooth_inverse>;
    // using VertexPatchType  = LevelVertexPatch<dim, fe_degree, Number>;

    MultigridSolver(
      const DoFHandler<dim>                            &dof_handler,
      const MGLevelObject<std::shared_ptr<MatrixFree>> &level_mfdata,
      const MGLevelObject<std::shared_ptr<MatrixFree>> &,
      const MGLevelObject<std::shared_ptr<CellPatchType>> &cell_data,
      const MGTransferCUDA<dim, Number>                   &transfer_dp,
      const Function<dim, Number>                         &boundary_values,
      const Function<dim, Number> &,
      std::shared_ptr<ConditionalOStream> pcout,
      const unsigned int                  n_cycles = 1)
      : dof_handler(&dof_handler)
      , transfer(&transfer_dp)
      , minlevel(1)
      , maxlevel(dof_handler.get_triangulation().n_global_levels() - 1)
      , n_cycles(n_cycles)
      , analytic_solution(boundary_values)
      , pcout(pcout)
    {
      AssertDimension(fe_degree, dof_handler.get_fe().degree);

      if (smooth_inverse == PSMF::SmootherVariant::MCS ||
          smooth_inverse == PSMF::SmootherVariant::MCS_CG ||
          smooth_inverse == PSMF::SmootherVariant::MCS_PCG ||
          smooth_inverse == PSMF::SmootherVariant::Chebyshev)
        minlevel = 0;

      matrix.resize(minlevel, maxlevel);

      for (unsigned int level = minlevel; level <= maxlevel; ++level)
        {
          matrix[level].initialize(level_mfdata[level], dof_handler, level);

          if (level == maxlevel)
            {
              matrix[level].initialize_dof_vector(solution);
              rhs = solution;
            }
        }

      // set up a mapping for the geometry representation
      MappingQ1<dim> mapping;

      // interpolate the inhomogeneous boundary conditions
      inhomogeneous_bc.clear();
      inhomogeneous_bc.resize(maxlevel + 1);

      {
        Timer time;
        // evaluate the right hand side in the equation, including the
        // residual from the inhomogeneous boundary conditions
        rhs = 0.;
        if (CT::SETS_ == "error_analysis")
          matrix[maxlevel].compute_rhs(rhs, solution);
        else
          rhs = 1.;

        *pcout << "RHS setup time:         " << time.wall_time() << "s"
               << std::endl;
      }

      if constexpr (smooth_inverse == PSMF::SmootherVariant::Chebyshev)
        {
          MGLevelObject<typename SmootherTypeCheb::AdditionalData>
            smoother_data;
          smoother_data.resize(minlevel, maxlevel);
          for (unsigned int level = minlevel; level <= maxlevel; ++level)
            {
              matrix[level].compute_diagonal();

              smoother_data[level].smoothing_range     = 20.;
              smoother_data[level].degree              = 5;
              smoother_data[level].eig_cg_n_iterations = 20;
              smoother_data[level].preconditioner =
                matrix[level].get_diagonal_inverse();
            }

          mg_smoother_cheb.initialize(matrix, smoother_data);
          mg_coarse_cheb.initialize(mg_smoother_cheb);

          mg_matrix.initialize(matrix);
          mg = std::make_unique<Multigrid<VectorType>>(mg_matrix,
                                                       mg_coarse_cheb,
                                                       transfer_dp,
                                                       mg_smoother_cheb,
                                                       mg_smoother_cheb,
                                                       minlevel,
                                                       maxlevel);

          preconditioner_mg = std::make_unique<
            PreconditionMG<dim, VectorType, MGTransferCUDA<dim, Number>>>(
            dof_handler, *mg, transfer_dp);
        }
      else if (smooth_inverse == PSMF::SmootherVariant::MCS ||
               smooth_inverse == PSMF::SmootherVariant::MCS_CG ||
               smooth_inverse == PSMF::SmootherVariant::MCS_PCG)
        {
          MGLevelObject<typename CellSmootherType::AdditionalData>
            smoother_data;
          smoother_data.resize(minlevel, maxlevel);
          for (unsigned int level = minlevel; level <= maxlevel; ++level)
            {
              smoother_data[level].data = cell_data[level];
            }

          mg_cell_smoother.initialize(matrix, smoother_data);
          mg_cell_coarse.initialize(mg_cell_smoother);

          mg_matrix.initialize(matrix);
          mg = std::make_unique<Multigrid<VectorType>>(mg_matrix,
                                                       mg_cell_coarse,
                                                       transfer_dp,
                                                       mg_cell_smoother,
                                                       mg_cell_smoother,
                                                       minlevel,
                                                       maxlevel);

          preconditioner_mg = std::make_unique<
            PreconditionMG<dim, VectorType, MGTransferCUDA<dim, Number>>>(
            dof_handler, *mg, transfer_dp);
        }
      else
        {
          AssertThrow(false, dealii::ExcMessage("Not implemented.\n"));
          // MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
          // smoother_data.resize(minlevel, maxlevel);
          // for (unsigned int level = minlevel; level <= maxlevel; ++level)
          //   {
          //     smoother_data[level].data = patch_data_dp[level];
          //   }
          //
          // mg_smoother.initialize(matrix, smoother_data);
          // mg_coarse.initialize(mg_smoother);
          //
          // mg_matrix.initialize(matrix);
          // mg = std::make_unique<Multigrid<VectorType>>(mg_matrix,
          //                                              mg_coarse,
          //                                              transfer_dp,
          //                                              mg_smoother,
          //                                              mg_smoother,
          //                                              minlevel,
          //                                              maxlevel);
          //
          // preconditioner_mg = std::make_unique<
          //   PreconditionMG<dim, VectorType, MGTransferCUDA<dim, Number>>>(
          //   dof_handler, *mg, transfer_dp);
        }

      // timers
      if (true)
        {
          all_mg_timers.resize((maxlevel - minlevel + 1));
          for (unsigned int i = 0; i < all_mg_timers.size(); ++i)
            all_mg_timers[i].resize(6);

          const auto create_mg_timer_function = [&](const unsigned int i,
                                                    const std::string &label) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            return [i, label, start, stop, this](const bool         flag,
                                                 const unsigned int level) {
              if (false && flag)
                std::cout << label << " " << level << std::endl;
              if (flag)
                {
                  cudaEventRecord(start);
                  // all_mg_timers[level - minlevel][i].second =
                  // std::chrono::system_clock::now();
                }
              else
                {
                  cudaEventRecord(stop);
                  cudaEventSynchronize(stop);
                  float milliseconds = 0;
                  cudaEventElapsedTime(&milliseconds, start, stop);
                  all_mg_timers[level - minlevel][i].first +=
                    milliseconds / 1e3;

                  // all_mg_timers[level - minlevel][i].first +=
                  //   std::chrono::duration_cast<std::chrono::nanoseconds>(
                  //     std::chrono::system_clock::now() -
                  //     all_mg_timers[level - minlevel][i].second)
                  //     .count() /
                  //   1e9;
                }
            };
          };

          {
            mg->connect_pre_smoother_step(
              create_mg_timer_function(0, "pre_smoother_step"));
            mg->connect_residual_step(
              create_mg_timer_function(1, "residual_step"));
            mg->connect_restriction(create_mg_timer_function(2, "restriction"));
            mg->connect_coarse_solve(
              create_mg_timer_function(3, "coarse_solve"));
            mg->connect_prolongation(
              create_mg_timer_function(4, "prolongation"));
            mg->connect_post_smoother_step(
              create_mg_timer_function(5, "post_smoother_step"));
          }

          all_mg_precon_timers.resize(2);

          const auto create_mg_precon_timer_function =
            [&](const unsigned int i) {
              cudaEvent_t start, stop;
              cudaEventCreate(&start);
              cudaEventCreate(&stop);

              return [i, start, stop, this](const bool flag) {
                if (flag)
                  cudaEventRecord(start);
                // all_mg_precon_timers[i].second =
                //   std::chrono::system_clock::now();
                else
                  {
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    float milliseconds = 0;
                    cudaEventElapsedTime(&milliseconds, start, stop);
                    all_mg_precon_timers[i].first += milliseconds / 1e3;

                    // all_mg_precon_timers[i].first +=
                    //   std::chrono::duration_cast<std::chrono::nanoseconds>(
                    //     std::chrono::system_clock::now() -
                    //     all_mg_precon_timers[i].second)
                    //     .count() /
                    //   1e9;
                  }
              };
            };

          preconditioner_mg->connect_transfer_to_mg(
            create_mg_precon_timer_function(0));
          preconditioner_mg->connect_transfer_to_global(
            create_mg_precon_timer_function(1));
        }

      size_t free_mem, total_mem;
      AssertCuda(cudaMemGetInfo(&free_mem, &total_mem));

      int mem_usage = (total_mem - free_mem) / 1024 / 1024;

      *pcout << "\nGPU Memory Usage [MB]: " << mem_usage << "\n";
    }

    std::vector<SolverData>
    static_comp()
    {
      *pcout << "Testing...\n";

      std::vector<SolverData> comp_data;

      std::string comp_name = "";

      const unsigned int n_dofs = dof_handler->n_dofs();
      const unsigned int n_mv   = n_dofs < 10000000 ? 100 : 20;

      auto tester = [&](auto kernel) {
        Timer              time;
        const unsigned int N         = 5;
        double             best_time = 1e10;
        for (unsigned int i = 0; i < N; ++i)
          {
            time.restart();
            for (unsigned int i = 0; i < n_mv; ++i)
              kernel(this);
            best_time = std::min(time.wall_time() / n_mv, best_time);
          }

        SolverData data;
        data.solver_name = comp_name;
        data.timing      = best_time;
        data.perf        = n_dofs / best_time;
        comp_data.push_back(data);
      };

      for (unsigned int s = 0; s < 2; ++s)
        {
          switch (s)
            {
              case 0:
                {
                  auto kernel = std::mem_fn(&MultigridSolver::do_matvec);
                  comp_name   = "Mat-vec";
                  tester(kernel);
                  break;
                }
              case 1:
                {
                  auto kernel = std::mem_fn(&MultigridSolver::do_smooth);
                  comp_name   = "Smooth";
                  tester(kernel);
                  break;
                }
              default:
                AssertThrow(false, ExcMessage("Invalid Solver Variant."));
            }
        }

      return comp_data;
    }

    // Return the solution vector for further processing
    const VectorType &
    get_solution()
    {
      return solution;
    }

    // Implement the vmult() function needed by the preconditioner interface
    void
    vmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
          const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
            &src) const
    {
      preconditioner_mg->vmult(dst, src);
    }



    void
    print_timings() const
    {
      {
        *pcout << " - #N of calls of multigrid: " << all_mg_counter << std::endl
               << std::endl;
        *pcout << " - Times of multigrid (levels):" << std::endl;

        const auto print_line = [&](const auto &vector) {
          for (const auto &i : vector)
            *pcout << std::scientific << std::setprecision(2) << std::setw(10)
                   << i.first;

          double sum = 0;

          for (const auto &i : vector)
            sum += i.first;

          *pcout << "   | " << std::scientific << std::setprecision(2)
                 << std::setw(10) << sum;

          *pcout << "\n";
        };

        for (unsigned int l = 0; l < all_mg_timers.size(); ++l)
          {
            *pcout << std::setw(4) << l << ": ";

            print_line(all_mg_timers[l]);
          }

        std::vector<
          std::pair<double, std::chrono::time_point<std::chrono::system_clock>>>
          sums(all_mg_timers[0].size());

        for (unsigned int i = 0; i < sums.size(); ++i)
          for (unsigned int j = 0; j < all_mg_timers.size(); ++j)
            sums[i].first += all_mg_timers[j][i].first;

        std::vector<
          std::pair<double, std::chrono::time_point<std::chrono::system_clock>>>
          lL(all_mg_timers[0].size());

        for (unsigned int i = 0; i < sums.size(); ++i)
          for (unsigned int j = 0; j < all_mg_timers.size() - 1; ++j)
            lL[i].first += all_mg_timers[j][i].first;

        *pcout
          << "   ------------------------------------------------------------------+-----------\n";
        *pcout << "sum:  ";
        print_line(sums);
        *pcout << "l<L:  ";
        print_line(lL);

        *pcout << std::endl;

        *pcout << " - Times of multigrid (solver <-> mg): ";

        for (const auto &i : all_mg_precon_timers)
          *pcout << i.first << " ";
        *pcout << std::endl;
        *pcout << std::endl;
      }
    }

    void
    clear_timings() const
    {
      for (auto &is : all_mg_timers)
        for (auto &i : is)
          i.first = 0.0;

      for (auto &i : all_mg_precon_timers)
        i.first = 0.0;

      all_mg_counter = 0;
    }


    // Solve with the conjugate gradient method preconditioned by the V-cycle
    // (invoking this->vmult) and return the number of iterations and the
    // reduction rate per GMRES iteration
    std::vector<SolverData>
    solve()
    {
      *pcout << "Solving in DP...\n";

      std::string solver_name = "GMRES";

      ReductionControl solver_control(CT::MAX_STEPS_, 1e-15, CT::REDUCE_);
      solver_control.enable_history_data();
      solver_control.log_history(true);

      // typename SolverFGMRES<VectorType>::AdditionalData additional_data(20);
      // SolverFGMRES<VectorType> solver(solver_control, additional_data);
      SolverGMRES<VectorType> solver(solver_control);

      Timer              time;
      const unsigned int N         = 5;
      double             best_time = 1e10;

      bool is_converged = true;
      try
        {
          {
            solution = 0;
            solver.solve(matrix[maxlevel], solution, rhs, *this);
            print_timings();
            clear_timings();
          }

          for (unsigned int i = 0; i < N; ++i)
            {
              time.reset();
              time.start();

              solution = 0;
              solver.solve(matrix[maxlevel], solution, rhs, *this);

              best_time = std::min(time.wall_time(), best_time);
            }
        }
      catch (...)
        {
          best_time = std::min(time.wall_time(), best_time);

          is_converged = false;

          *pcout << "\n!!! Solver not Converged within " << CT::MAX_STEPS_
                 << " steps. !!!\n\n";
        }

      auto n_iter     = solver_control.last_step();
      auto residual_0 = solver_control.initial_value();
      auto residual_n = solver_control.last_value();
      auto reduction  = solver_control.reduction();

      // *** average reduction: r_n = rho^n * r_0
      const double rho =
        std::pow(residual_n / residual_0, static_cast<double>(1. / n_iter));
      const double convergence_rate =
        1. / n_iter * std::log10(residual_0 / residual_n);

      const auto n_step = -10 * std::log10(rho);
      const auto n_frac = std::log(reduction) / std::log(rho);

      size_t free_mem, total_mem;
      AssertCuda(cudaMemGetInfo(&free_mem, &total_mem));

      int mem_usage = (total_mem - free_mem) / 1024 / 1024;

      SolverData data;
      data.solver_name      = solver_name;
      data.n_iteration      = n_iter;
      data.n_step           = n_frac;
      data.residual         = residual_n;
      data.reduction_rate   = rho;
      data.convergence_rate = convergence_rate;
      data.timing           = best_time;
      data.mem_usage        = mem_usage;

      if (smooth_inverse == PSMF::SmootherVariant::MCS_CG ||
          smooth_inverse == PSMF::SmootherVariant::MCS_PCG)
        {
          auto vec = mg_cell_smoother.smoothers[maxlevel].get_cg_solver_info();

          data.cg_it    = vec[0];
          data.cg_error = vec[1];
        }

      solver_data.push_back(data);

      if (is_converged)
        {
          auto history_data = solver_control.get_history_data();
          for (auto i = 1U; i < n_iter + 1; ++i)
            *pcout << "step " << i << ": " << history_data[i] / residual_0
                   << "\n";
        }

      return solver_data;
    }

    // run smooth in double precision
    void
    do_smooth()
    {
      if constexpr (smooth_inverse == PSMF::SmootherVariant::Chebyshev)
        (mg_smoother_cheb).smooth(maxlevel, solution, rhs);
      else if (smooth_inverse == PSMF::SmootherVariant::MCS ||
               smooth_inverse == PSMF::SmootherVariant::MCS_CG ||
               smooth_inverse == PSMF::SmootherVariant::MCS_PCG)
        (mg_cell_smoother).smooth(maxlevel, solution, rhs);
      else
        AssertThrow(false, dealii::ExcMessage("Not implemented.\n"));
      cudaDeviceSynchronize();

      // solution.print(std::cout);
      // std::cout << solution.l2_norm() << std::endl;
      //
      // AssertThrow(false, dealii::ExcMessage("debug."));
    }

    // run matrix-vector product in double precision
    void
    do_matvec()
    {
      matrix[maxlevel].vmult(solution, rhs);
      cudaDeviceSynchronize();

      // std::cout << rhs.l2_norm() << " " << solution.l2_norm() << std::endl;

      // AssertThrow(false, dealii::ExcMessage("debug."));
      // for (unsigned int i = 0; i < rhs.size(); ++i)
      //   {
      //     LinearAlgebra::ReadWriteVector<double> rw_vector(rhs.size());
      //
      //     // for (unsigned int i = 0; i < rhs.size(); ++i)
      //     rw_vector[i] = 1. + 0;
      //
      //     rhs.import(rw_vector, VectorOperation::insert);
      //
      //     matrix[maxlevel].vmult(solution, rhs);
      //
      //     solution.print(std::cout);
      //     std::cout << solution.l2_norm() << std::endl;
      //
      //     // if (i == 0)
      //     break;
      //   }
      //
      // AssertThrow(false, dealii::ExcMessage("debug."));
    }

  private:
    const SmartPointer<const DoFHandler<dim>>             dof_handler;
    const SmartPointer<const MGTransferCUDA<dim, Number>> transfer;

    MGLevelObject<MatrixType> matrix;

    std::vector<std::map<unsigned int, Number>> inhomogeneous_bc;

    /**
     * Lowest level of cells.
     */
    unsigned int minlevel;

    /**
     * Highest level of cells.
     */
    unsigned int maxlevel;

    /**
     * The solution vector
     */
    mutable VectorType solution;

    /**
     * Original right hand side vector
     */
    mutable VectorType rhs;

    // MGSmootherPrecondition<MatrixType, SmootherType, VectorType> mg_smoother;

    mutable MGSmootherPrecondition<MatrixType, SmootherTypeCheb, VectorType>
      mg_smoother_cheb;

    MGSmootherPrecondition<MatrixType, CellSmootherType, VectorType>
      mg_cell_smoother;

    /**
     * The coarse solver
     */
    // MGCoarseGridApplySmoother<VectorType> mg_coarse;

    mutable MGCoarseGridApplySmoother<VectorType> mg_coarse_cheb;

    mutable MGCoarseGridApplySmoother<VectorType> mg_cell_coarse;

    mutable std::unique_ptr<
      PreconditionMG<dim, VectorType, MGTransferCUDA<dim, Number>>>
      preconditioner_mg;

    mutable mg::Matrix<VectorType> mg_matrix;

    mutable std::unique_ptr<Multigrid<VectorType>> mg;

    /**
     * Number of cycles to be done in the FMG cycle
     */
    const unsigned int n_cycles;

    /**
     * Collection of compute times on various levels
     */
    mutable std::vector<SolverData> solver_data;

    /**
     * Function for boundary values that we keep as analytic
     * solution
     */
    const Function<dim, Number> &analytic_solution;

    std::shared_ptr<ConditionalOStream> pcout;

    mutable unsigned int all_mg_counter = 0;

    mutable std::vector<std::vector<
      std::pair<double, std::chrono::time_point<std::chrono::system_clock>>>>
      all_mg_timers;

    mutable std::vector<
      std::pair<double, std::chrono::time_point<std::chrono::system_clock>>>
      all_mg_precon_timers;
  };


} // namespace PSMF

#endif // SOLVER_CUH
