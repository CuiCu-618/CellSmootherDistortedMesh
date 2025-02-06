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
#include <iomanip>

#include "cuda_mg_transfer.cuh"
#include "laplace_operator.cuh"
#include "patch_base.cuh"
#include "patch_smoother.cuh"

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

  /**
   * @brief Multigrid Method with vertex-patch smoother.
   *
   * @tparam dim
   * @tparam fe_degree
   * @tparam Number full number
   * @tparam smooth_kernel
   * @tparam Number1 vcycle number
   */
  template <int dim,
            int fe_degree,
            typename Number,
            LocalSolverVariant local_solver,
            LaplaceVariant     lapalace_kernel,
            LaplaceVariant     smooth_vmult,
            SmootherVariant    smooth_inverse,
            typename Number2>
  class MultigridSolver
  {
  public:
    using VectorType =
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>;
    using VectorType2 =
      LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA>;
    using MatrixType = LaplaceOperator<dim, fe_degree, Number, lapalace_kernel>;
    using MatrixType2 =
      LaplaceOperator<dim, fe_degree, Number2, lapalace_kernel>;
    using SmootherType       = PatchSmoother<MatrixType2,
                                             dim,
                                             fe_degree,
                                             local_solver,
                                             smooth_vmult,
                                             smooth_inverse>;
    using SmootherTypeCoarse = PatchSmoother<MatrixType2,
                                             dim,
                                             fe_degree,
                                             LocalSolverVariant::Exact,
                                             smooth_vmult,
                                             smooth_inverse>;
    using MatrixFreeType     = LevelVertexPatch<dim, fe_degree, Number>;
    using MatrixFreeType2    = LevelVertexPatch<dim, fe_degree, Number2>;

    MultigridSolver(
      const DoFHandler<dim>                                 &dof_handler,
      const MGLevelObject<std::shared_ptr<MatrixFreeType>>  &mfdata_dp,
      const MGLevelObject<std::shared_ptr<MatrixFreeType2>> &mfdata,
      const MGTransferCUDA<dim, Number2>                    &transfer,
      const Function<dim, Number>                           &boundary_values,
      const Function<dim, Number>                           &right_hand_side,
      std::shared_ptr<ConditionalOStream>                    pcout,
      const unsigned int                                     n_cycles = 1)
      : dof_handler(&dof_handler)
      , transfer(&transfer)
      , minlevel(1)
      , maxlevel(dof_handler.get_triangulation().n_global_levels() - 1)
      , solution(minlevel, maxlevel)
      , rhs(minlevel, maxlevel)
      , residual(minlevel, maxlevel)
      , defect(minlevel, maxlevel)
      , t(minlevel, maxlevel)
      , solution_update(minlevel, maxlevel)
      , n_cycles(n_cycles)
      , timings(maxlevel + 1)
      , analytic_solution(boundary_values)
      , pcout(pcout)
    {
      AssertDimension(fe_degree, dof_handler.get_fe().degree);

      matrix_dp.resize(minlevel, maxlevel);
      matrix.resize(minlevel, maxlevel);

      for (unsigned int level = minlevel; level <= maxlevel; ++level)
        {
          matrix_dp[level].initialize(mfdata_dp[level], dof_handler, level);
          matrix[level].initialize(mfdata[level], dof_handler, level);

          matrix[level].initialize_dof_vector(defect[level]);
          t[level].reinit(defect[level]);
          solution_update[level].reinit(defect[level]);

          if (level == maxlevel)
            {
              matrix_dp[level].initialize_dof_vector(solution[level]);
              rhs[level].reinit(solution[level]);
              residual[level].reinit(solution[level]);
            }
        }

      // set up a mapping for the geometry representation
      MappingQ1<dim> mapping;

      // interpolate the inhomogeneous boundary conditions
      inhomogeneous_bc.clear();
      inhomogeneous_bc.resize(maxlevel + 1);
      if (CT::SETS_ == "error_analysis")
        for (unsigned int level = minlevel; level <= maxlevel; ++level)
          {
            // Quadrature<dim - 1> face_quad(
            //   dof_handler.get_fe().get_unit_face_support_points());
            // FEFaceValues<dim>                    fe_values(mapping,
            //                             dof_handler.get_fe(),
            //                             face_quad,
            //                             update_quadrature_points);
            // std::vector<types::global_dof_index> face_dof_indices(
            //   dof_handler.get_fe().dofs_per_face);

            // typename DoFHandler<dim>::cell_iterator cell =
            //                                           dof_handler.begin(level),
            //                                         endc =
            //                                           dof_handler.end(level);
            // for (; cell != endc; ++cell)
            //   if (cell->level_subdomain_id() !=
            //       numbers::artificial_subdomain_id)
            //     for (unsigned int face_no = 0;
            //          face_no < GeometryInfo<dim>::faces_per_cell;
            //          ++face_no)
            //       if (cell->at_boundary(face_no))
            //         {
            //           const typename DoFHandler<dim>::face_iterator face =
            //             cell->face(face_no);
            //           face->get_mg_dof_indices(level, face_dof_indices);
            //           fe_values.reinit(cell, face_no);
            //           for (unsigned int i = 0; i < face_dof_indices.size();
            //           ++i)
            //             if
            //             (dof_handler.locally_owned_mg_dofs(level).is_element(
            //                   face_dof_indices[i]))
            //               {
            //                 const double value = analytic_solution.value(
            //                   fe_values.quadrature_point(i));
            //                 if (value != 0.0)
            //                   inhomogeneous_bc[level][face_dof_indices[i]] =
            //                     value;
            //               }
            //         }
          }


      {
        // evaluate the right hand side in the equation, including the
        // residual from the inhomogeneous boundary conditions
        // set_inhomogeneous_bc<false>(maxlevel);
        rhs[maxlevel] = 0.;
        if (CT::SETS_ == "error_analysis")
          matrix_dp[maxlevel].compute_residual(rhs[maxlevel],
                                               solution[maxlevel],
                                               right_hand_side,
                                               boundary_values,
                                               maxlevel);
        else
          rhs[maxlevel] = 1.;
      }


      {
        MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
        MGLevelObject<typename SmootherTypeCoarse::AdditionalData>
          smoother_data_coarse;
        smoother_data.resize(minlevel, maxlevel);
        smoother_data_coarse.resize(minlevel, maxlevel);
        for (unsigned int level = minlevel; level <= maxlevel; ++level)
          {
            smoother_data[level].data         = mfdata[level];
            smoother_data[level].n_iterations = CT::N_SMOOTH_STEPS_;

            smoother_data_coarse[level].data         = mfdata[level];
            smoother_data_coarse[level].n_iterations = CT::N_SMOOTH_STEPS_;
          }

        mg_smoother.initialize(matrix, smoother_data);
        mg_smoother_coarse.initialize(matrix, smoother_data_coarse);
        mg_coarse.initialize(mg_smoother_coarse);
      }
    }


    std::vector<SolverData>
    static_comp()
    {
      *pcout << "Testing...\n";

      std::vector<SolverData> comp_data;

      std::string comp_name = "";

      const unsigned int n_dofs = dof_handler->n_dofs();
      const unsigned int n_mv   = n_dofs < 10000000 ? 20 : 4;

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

      for (unsigned int s = 0; s < 3; ++s)
        {
          switch (s)
            {
              case 0:
                {
                  auto kernel = std::mem_fn(&MultigridSolver::do_matvec);
                  comp_name   = "Mat-vec DP";
                  tester(kernel);
                  break;
                }
              case 1:
                {
                  auto kernel =
                    std::mem_fn(&MultigridSolver::do_matvec_smoother);
                  comp_name = "Mat-vec SP";
                  tester(kernel);
                  break;
                }
              case 2:
                {
                  auto kernel = std::mem_fn(&MultigridSolver::do_smoother);
                  comp_name   = "Smooth";
                  tester(kernel);
                  break;
                }
              case 3:
                {
                  auto kernel = std::mem_fn(&MultigridSolver::do_vcycle);
                  comp_name   = "V-cycle";
                  tester(kernel);
                  break;
                }
              default:
                AssertThrow(false, ExcMessage("Invalid Solver Variant."));
            }
        }


      return comp_data;
    }


    // Implement the vmult() function needed by the preconditioner interface
    void
    vmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
          const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
            &src) const
    {
      all_mg_counter++;

      preconditioner_mg->vmult(dst, src);
    }

    // Return the solution vector for further processing
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &
    get_solution()
    {
      // set_inhomogeneous_bc<false>(maxlevel);
      return solution[maxlevel];
    }

    void
    print_timings() const
    {
      // if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
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

        *pcout
          << "   ----------------------------------------------------------------------------+-----------\n";
        *pcout << "      ";
        print_line(sums);

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
      *pcout << "Solving...\n";

      return solver_data;

      std::string solver_name = "GMRES";

      mg::Matrix<VectorType2> mg_matrix(matrix);

      Multigrid<VectorType2> mg(mg_matrix,
                                mg_coarse,
                                *transfer,
                                mg_smoother,
                                mg_smoother,
                                minlevel,
                                maxlevel);

      preconditioner_mg = std::make_unique<
        PreconditionMG<dim, VectorType2, MGTransferCUDA<dim, Number2>>>(
        *dof_handler, mg, *transfer);

      // timers
      if (true)
        {
          all_mg_timers.resize((maxlevel - minlevel + 1));
          for (unsigned int i = 0; i < all_mg_timers.size(); ++i)
            all_mg_timers[i].resize(7);

          const auto create_mg_timer_function = [&](const unsigned int i,
                                                    const std::string &label) {
            return [i, label, this](const bool flag, const unsigned int level) {
              if (false && flag)
                std::cout << label << " " << level << std::endl;
              if (flag)
                all_mg_timers[level - minlevel][i].second =
                  std::chrono::system_clock::now();
              else
                all_mg_timers[level - minlevel][i].first +=
                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now() -
                    all_mg_timers[level - minlevel][i].second)
                    .count() /
                  1e9;
            };
          };

          {
            mg.connect_pre_smoother_step(
              create_mg_timer_function(0, "pre_smoother_step"));
            mg.connect_residual_step(
              create_mg_timer_function(1, "residual_step"));
            mg.connect_restriction(create_mg_timer_function(2, "restriction"));
            mg.connect_coarse_solve(
              create_mg_timer_function(3, "coarse_solve"));
            mg.connect_prolongation(
              create_mg_timer_function(4, "prolongation"));
            mg.connect_edge_prolongation(
              create_mg_timer_function(5, "edge_prolongation"));
            mg.connect_post_smoother_step(
              create_mg_timer_function(6, "post_smoother_step"));
          }

          all_mg_precon_timers.resize(2);

          const auto create_mg_precon_timer_function =
            [&](const unsigned int i) {
              return [i, this](const bool flag) {
                if (flag)
                  all_mg_precon_timers[i].second =
                    std::chrono::system_clock::now();
                else
                  all_mg_precon_timers[i].first +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::system_clock::now() -
                      all_mg_precon_timers[i].second)
                      .count() /
                    1e9;
              };
            };

          preconditioner_mg->connect_transfer_to_mg(
            create_mg_precon_timer_function(0));
          preconditioner_mg->connect_transfer_to_global(
            create_mg_precon_timer_function(1));
        }

      ReductionControl solver_control(CT::MAX_STEPS_, 1e-14, CT::REDUCE_);
      solver_control.enable_history_data();
      solver_control.log_history(true);

#if SCHWARZTYPE == 0
      SolverGMRES<VectorType> solver(solver_control);
      // SolverFGMRES<VectorType> solver(solver_control);
#else
      SolverCG<VectorType> solver(solver_control);
#endif

      Timer              time;
      const unsigned int N         = 5;
      double             best_time = 1e10;

      bool is_converged = true;
      try
        {
          {
            solution[maxlevel] = 0;
            solver.solve(matrix_dp[maxlevel],
                         solution[maxlevel],
                         rhs[maxlevel],
                         *this);
            print_timings();
            clear_timings();
          }

          for (unsigned int i = 0; i < N; ++i)
            {
              time.reset();
              time.start();

              solution[maxlevel] = 0;
              solver.solve(matrix_dp[maxlevel],
                           solution[maxlevel],
                           rhs[maxlevel],
                           *this);

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

    // run matrix-vector product in double precision
    void
    do_matvec()
    {
      matrix_dp[maxlevel].vmult(residual[maxlevel], solution[maxlevel]);
      cudaDeviceSynchronize();
    }

    // run matrix-vector product in single precision
    void
    do_matvec_smoother()
    {
      matrix[maxlevel].vmult(solution_update[maxlevel], defect[maxlevel]);
      cudaDeviceSynchronize();
    }

    // run smoother in single precision
    void
    do_smoother()
    {
      (mg_smoother)
        .smooth(maxlevel, solution_update[maxlevel], defect[maxlevel]);
      cudaDeviceSynchronize();
    }

    // run v-cycle in single precision
    void
    do_vcycle()
    {
      v_cycle(maxlevel, 1);
    }

  private:
    template <bool is_zero = false>
    void
    set_inhomogeneous_bc(const unsigned int level)
    {
      unsigned int n_inhomogeneous_bc = inhomogeneous_bc[level].size();
      if (n_inhomogeneous_bc != 0)
        {
          const unsigned int block_size = 256;
          const unsigned int inhomogeneous_n_blocks =
            std::ceil(static_cast<double>(n_inhomogeneous_bc) /
                      static_cast<double>(block_size));
          const unsigned int inhomogeneous_x_n_blocks =
            std::round(std::sqrt(inhomogeneous_n_blocks));
          const unsigned int inhomogeneous_y_n_blocks =
            std::ceil(static_cast<double>(inhomogeneous_n_blocks) /
                      static_cast<double>(inhomogeneous_x_n_blocks));

          dim3 inhomogeneous_grid_dim(inhomogeneous_x_n_blocks,
                                      inhomogeneous_y_n_blocks);
          dim3 inhomogeneous_block_dim(block_size);

          std::vector<unsigned int> inhomogeneous_index_host(
            n_inhomogeneous_bc);
          std::vector<Number> inhomogeneous_value_host(n_inhomogeneous_bc);
          unsigned int        count = 0;
          for (auto &i : inhomogeneous_bc[level])
            {
              inhomogeneous_index_host[count] = i.first;
              inhomogeneous_value_host[count] = i.second;
              count++;
            }

          unsigned int *inhomogeneous_index;
          Number       *inhomogeneous_value;

          cudaError_t cuda_error =
            cudaMalloc(&inhomogeneous_index,
                       n_inhomogeneous_bc * sizeof(unsigned int));
          AssertCuda(cuda_error);

          cuda_error = cudaMemcpy(inhomogeneous_index,
                                  inhomogeneous_index_host.data(),
                                  n_inhomogeneous_bc * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice);
          AssertCuda(cuda_error);

          cuda_error = cudaMalloc(&inhomogeneous_value,
                                  n_inhomogeneous_bc * sizeof(Number2));
          AssertCuda(cuda_error);

          cuda_error = cudaMemcpy(inhomogeneous_value,
                                  inhomogeneous_value_host.data(),
                                  n_inhomogeneous_bc * sizeof(Number2),
                                  cudaMemcpyHostToDevice);
          AssertCuda(cuda_error);


          set_inhomogeneous_dofs<is_zero, Number>
            <<<inhomogeneous_grid_dim, inhomogeneous_block_dim>>>(
              inhomogeneous_index,
              inhomogeneous_value,
              n_inhomogeneous_bc,
              solution[level].get_values());
          AssertCudaKernel();
        }
    }

    template <bool is_d2f = true, typename number, typename number2>
    void
    convert_precision(
      LinearAlgebra::distributed::Vector<number, MemorySpace::CUDA>        &dst,
      const LinearAlgebra::distributed::Vector<number2, MemorySpace::CUDA> &src)
      const
    {
      unsigned int n_dofs = dst.size();
      if (n_dofs != 0)
        {
          const unsigned int block_size = 256;
          const unsigned int n_blocks   = std::ceil(
            static_cast<double>(n_dofs) / static_cast<double>(block_size));
          const unsigned int x_n_blocks = std::round(std::sqrt(n_blocks));
          const unsigned int y_n_blocks = std::ceil(
            static_cast<double>(n_blocks) / static_cast<double>(x_n_blocks));

          dim3 grid_dim(x_n_blocks, y_n_blocks);
          dim3 block_dim(block_size);
          copy_vector<is_d2f, number, number2>
            <<<grid_dim, block_dim>>>(dst.get_values(),
                                      src.get_values(),
                                      n_dofs);
          AssertCudaKernel();
        }
    }

    // Implement the V-cycle
    void
    v_cycle(const unsigned int level, const unsigned int my_n_cycles) const
    {
      cudaEvent_t start, stop;
      AssertCuda(cudaEventCreate(&start));
      AssertCuda(cudaEventCreate(&stop));
      float gpu_time = 0.0f;

      if (level == minlevel)
        {
          // Timer time;
          cudaEventRecord(start);
          (mg_coarse)(level, solution_update[level], defect[level]);
          cudaEventRecord(stop);
          cudaDeviceSynchronize();
          AssertCuda(cudaEventElapsedTime(&gpu_time, start, stop));
          timings[level][0] += gpu_time;
          timings[level][1] += 1;
          return;
        }

      for (unsigned int c = 0; c < my_n_cycles; ++c)
        {
          // Timer time;
          cudaEventRecord(start);
          if (c == 0)
            (mg_smoother).apply(level, solution_update[level], defect[level]);
          else
            (mg_smoother).smooth(level, solution_update[level], defect[level]);
          cudaEventRecord(stop);
          cudaDeviceSynchronize();
          AssertCuda(cudaEventElapsedTime(&gpu_time, start, stop));
          timings[level][5] += gpu_time;

          cudaEventRecord(start);
          matrix[level].vmult(t[level], solution_update[level]);
          cudaEventRecord(stop);
          cudaDeviceSynchronize();
          AssertCuda(cudaEventElapsedTime(&gpu_time, start, stop));
          timings[level][0] += gpu_time;

          cudaEventRecord(start);
          t[level].sadd(-1.0, 1.0, defect[level]);
          timings[level][4] += gpu_time;

          cudaEventRecord(start);
          defect[level - 1] = 0;
          transfer->restrict_and_add(level, defect[level - 1], t[level]);
          cudaEventRecord(stop);
          cudaDeviceSynchronize();
          AssertCuda(cudaEventElapsedTime(&gpu_time, start, stop));
          timings[level][1] += gpu_time;

          v_cycle(level - 1, 1);

          cudaEventRecord(start);
          transfer->prolongate_and_add(level,
                                       solution_update[level],
                                       solution_update[level - 1]);
          cudaEventRecord(stop);
          cudaDeviceSynchronize();
          AssertCuda(cudaEventElapsedTime(&gpu_time, start, stop));
          timings[level][2] += gpu_time;

          cudaEventRecord(start);
          // solution_update[level] += t[level];
          cudaEventRecord(stop);
          cudaDeviceSynchronize();
          AssertCuda(cudaEventElapsedTime(&gpu_time, start, stop));
          timings[level][4] += gpu_time;

          cudaEventRecord(start);
          (mg_smoother).smooth(level, solution_update[level], defect[level]);
          cudaEventRecord(stop);
          cudaDeviceSynchronize();
          AssertCuda(cudaEventElapsedTime(&gpu_time, start, stop));
          timings[level][5] += gpu_time;
        }
      AssertCuda(cudaEventDestroy(start));
      AssertCuda(cudaEventDestroy(stop));
    }

    const SmartPointer<const DoFHandler<dim>> dof_handler;

    const SmartPointer<const MGTransferCUDA<dim, Number2>> transfer;

    std::vector<std::map<unsigned int, Number>> inhomogeneous_bc;

    MGLevelObject<MatrixType>  matrix_dp;
    MGLevelObject<MatrixType2> matrix;

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
    mutable MGLevelObject<VectorType> solution;

    /**
     * Original right hand side vector
     */
    mutable MGLevelObject<VectorType> rhs;

    /**
     * Residual vector before it is passed down into float through the
     * v-cycle
     */
    mutable MGLevelObject<VectorType> residual;

    /**
     * Input vector for the cycle. Contains the defect of the outer method
     * projected to the multilevel vectors.
     */
    mutable MGLevelObject<VectorType2> defect;

    /**
     * Auxiliary vector.
     */
    mutable MGLevelObject<VectorType2> t;

    /**
     * Auxiliary vector for the solution update
     */
    mutable MGLevelObject<VectorType2> solution_update;

    // MGLevelObject<SmootherType> smooth;

    MGSmootherPrecondition<MatrixType2, SmootherType, VectorType2> mg_smoother;

    /**
     * The coarse solver
     */

    MGSmootherPrecondition<MatrixType2, SmootherTypeCoarse, VectorType2>
      mg_smoother_coarse;

    MGCoarseGridApplySmoother<VectorType2> mg_coarse;

    // MGCoarseFromSmoother<VectorType2, MGLevelObject<SmootherType>> coarse;

    /**
     * Number of cycles to be done in the FMG cycle
     */
    const unsigned int n_cycles;

    /**
     * Collection of compute times on various levels
     */
    mutable std::vector<std::array<double, 6>> timings;

    /**
     * Collection of compute times on various levels
     */
    mutable std::vector<SolverData> solver_data;

    /**
     * Function for boundary values that we keep as analytic solution
     */
    const Function<dim, Number> &analytic_solution;

    std::shared_ptr<ConditionalOStream> pcout;

    mutable std::unique_ptr<
      PreconditionMG<dim, VectorType2, MGTransferCUDA<dim, Number2>>>
      preconditioner_mg;

    mutable unsigned int all_mg_counter = 0;

    mutable std::vector<std::vector<
      std::pair<double, std::chrono::time_point<std::chrono::system_clock>>>>
      all_mg_timers;

    mutable std::vector<
      std::pair<double, std::chrono::time_point<std::chrono::system_clock>>>
      all_mg_precon_timers;
  };



  template <int dim,
            int fe_degree,
            typename Number,
            LocalSolverVariant local_solver,
            LaplaceVariant     lapalace_kernel,
            LaplaceVariant     smooth_vmult,
            SmootherVariant    smooth_inverse>
  class MultigridSolver<dim,
                        fe_degree,
                        Number,
                        local_solver,
                        lapalace_kernel,
                        smooth_vmult,
                        smooth_inverse,
                        Number>
  {
  public:
    using VectorType =
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>;
    using MatrixType = LaplaceOperator<dim, fe_degree, Number, lapalace_kernel>;
    using SmootherType       = PatchSmoother<MatrixType,
                                             dim,
                                             fe_degree,
                                             local_solver,
                                             smooth_vmult,
                                             smooth_inverse>;
    using SmootherTypeCoarse = PatchSmoother<MatrixType,
                                             dim,
                                             fe_degree,
                                             LocalSolverVariant::Exact,
                                             smooth_vmult,
                                             smooth_inverse>;
    using MatrixFreeType     = LevelVertexPatch<dim, fe_degree, Number>;

    MultigridSolver(
      const DoFHandler<dim>                                &dof_handler,
      const MGLevelObject<std::shared_ptr<MatrixFreeType>> &mfdata_dp,
      const MGLevelObject<std::shared_ptr<MatrixFreeType>> &,
      const MGTransferCUDA<dim, Number>  &transfer_dp,
      const Function<dim, Number>        &boundary_values,
      const Function<dim, Number>        &right_hand_side,
      std::shared_ptr<ConditionalOStream> pcout,
      const unsigned int                  n_cycles = 1)
      : dof_handler(&dof_handler)
      , transfer(&transfer_dp)
      , minlevel(1)
      , maxlevel(dof_handler.get_triangulation().n_global_levels() - 1)
      , solution(minlevel, maxlevel)
      , rhs(minlevel, maxlevel)
      , defect(minlevel, maxlevel)
      , t(minlevel, maxlevel)
      , n_cycles(n_cycles)
      , analytic_solution(boundary_values)
      , pcout(pcout)
    {
      AssertDimension(fe_degree, dof_handler.get_fe().degree);

      matrix.resize(minlevel, maxlevel);

      for (unsigned int level = minlevel; level <= maxlevel; ++level)
        {
          matrix[level].initialize(mfdata_dp[level], dof_handler, level);

          matrix[level].initialize_dof_vector(solution[level]);
          defect[level] = solution[level];
          rhs[level]    = solution[level];
          t[level]      = solution[level];
        }

      // set up a mapping for the geometry representation
      MappingQ1<dim> mapping;

      // interpolate the inhomogeneous boundary conditions
      inhomogeneous_bc.clear();
      inhomogeneous_bc.resize(maxlevel + 1);
      if (CT::SETS_ == "error_analysis")
        for (unsigned int level = minlevel; level <= maxlevel; ++level)
          {
            // Quadrature<dim - 1> face_quad(
            //   dof_handler.get_fe().get_unit_face_support_points());
            // FEFaceValues<dim>                    fe_values(mapping,
            //                             dof_handler.get_fe(),
            //                             face_quad,
            //                             update_quadrature_points);
            // std::vector<types::global_dof_index> face_dof_indices(
            //   dof_handler.get_fe().dofs_per_face);

            // typename DoFHandler<dim>::cell_iterator cell =
            //                                           dof_handler.begin(level),
            //                                         endc =
            //                                           dof_handler.end(level);
            // for (; cell != endc; ++cell)
            //   if (cell->level_subdomain_id() !=
            //       numbers::artificial_subdomain_id)
            //     for (unsigned int face_no = 0;
            //          face_no < GeometryInfo<dim>::faces_per_cell;
            //          ++face_no)
            //       if (cell->at_boundary(face_no))
            //         {
            //           const typename DoFHandler<dim>::face_iterator face =
            //             cell->face(face_no);
            //           face->get_mg_dof_indices(level, face_dof_indices);
            //           fe_values.reinit(cell, face_no);
            //           for (unsigned int i = 0; i < face_dof_indices.size();
            //           ++i)
            //             if
            //             (dof_handler.locally_owned_mg_dofs(level).is_element(
            //                   face_dof_indices[i]))
            //               {
            //                 const double value = analytic_solution.value(
            //                   fe_values.quadrature_point(i));
            //                 if (value != 0.0)
            //                   inhomogeneous_bc[level][face_dof_indices[i]] =
            //                     value;
            //               }
            //         }
          }

      for (int level = maxlevel; level >= minlevel; --level)
        {
          // evaluate the right hand side in the equation, including the
          // residual from the inhomogeneous boundary conditions
          // set_inhomogeneous_bc<false>(level);
          rhs[level] = 0.;
          if (level == maxlevel)
            if (CT::SETS_ == "error_analysis")
              matrix[level].compute_residual(rhs[level],
                                             solution[level],
                                             right_hand_side,
                                             boundary_values,
                                             level);
            else
              rhs[level] = 1.;
          else
            transfer_dp.restrict_and_add(level + 1, rhs[level], rhs[level + 1]);
        }

      {
        MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
        MGLevelObject<typename SmootherTypeCoarse::AdditionalData>
          smoother_data_coarse;
        smoother_data.resize(minlevel, maxlevel);
        smoother_data_coarse.resize(minlevel, maxlevel);
        for (unsigned int level = minlevel; level <= maxlevel; ++level)
          {
            smoother_data[level].data         = mfdata_dp[level];
            smoother_data[level].n_iterations = CT::N_SMOOTH_STEPS_;

            smoother_data_coarse[level].data         = mfdata_dp[level];
            smoother_data_coarse[level].n_iterations = CT::N_SMOOTH_STEPS_;
          }

        mg_smoother.initialize(matrix, smoother_data);
        mg_smoother_coarse.initialize(matrix, smoother_data_coarse);
        mg_coarse.initialize(mg_smoother_coarse);
      }
    }

    std::vector<SolverData>
    static_comp()
    {
      *pcout << "Testing...\n";

      std::vector<SolverData> comp_data;

      std::string comp_name = "";

      const unsigned int n_dofs = dof_handler->n_dofs();
      const unsigned int n_mv   = n_dofs < 10000000 ? 10 : 4;

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
              case 2:
                {
                  auto kernel = std::mem_fn(&MultigridSolver::do_vcycle);
                  comp_name   = "V-cycle";
                  tester(kernel);
                  break;
                }
              case 3:
                {
                  auto kernel = std::mem_fn(&MultigridSolver::do_fmgcycle);
                  comp_name   = "FMG-cycle";
                  tester(kernel);
                  break;
                }
              default:
                AssertThrow(false, ExcMessage("Invalid Solver Variant."));
            }
        }

      return comp_data;
    }


    // Implement the vmult() function needed by the preconditioner interface
    void
    vmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
          const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
            &src) const
    {
      all_mg_counter++;

      preconditioner_mg->vmult(dst, src);
    }

    // Return the solution vector for further processing
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &
    get_solution()
    {
      // set_inhomogeneous_bc<false>(maxlevel);
      return solution[maxlevel];
    }

    void
    print_timings() const
    {
      // if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
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

        *pcout
          << "   ----------------------------------------------------------------------------+-----------\n";
        *pcout << "      ";
        print_line(sums);

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
      *pcout << "Solving...\n";

      return solver_data;

      mg::Matrix<VectorType> mg_matrix(matrix);

      Multigrid<VectorType> mg(mg_matrix,
                               mg_coarse,
                               *transfer,
                               mg_smoother,
                               mg_smoother,
                               minlevel,
                               maxlevel);

      preconditioner_mg = std::make_unique<
        PreconditionMG<dim, VectorType, MGTransferCUDA<dim, Number>>>(
        *dof_handler, mg, *transfer);

      // timers
      if (true)
        {
          all_mg_timers.resize((maxlevel - minlevel + 1));
          for (unsigned int i = 0; i < all_mg_timers.size(); ++i)
            all_mg_timers[i].resize(7);

          const auto create_mg_timer_function = [&](const unsigned int i,
                                                    const std::string &label) {
            return [i, label, this](const bool flag, const unsigned int level) {
              if (false && flag)
                std::cout << label << " " << level << std::endl;
              if (flag)
                all_mg_timers[level - minlevel][i].second =
                  std::chrono::system_clock::now();
              else
                all_mg_timers[level - minlevel][i].first +=
                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now() -
                    all_mg_timers[level - minlevel][i].second)
                    .count() /
                  1e9;
            };
          };

          {
            mg.connect_pre_smoother_step(
              create_mg_timer_function(0, "pre_smoother_step"));
            mg.connect_residual_step(
              create_mg_timer_function(1, "residual_step"));
            mg.connect_restriction(create_mg_timer_function(2, "restriction"));
            mg.connect_coarse_solve(
              create_mg_timer_function(3, "coarse_solve"));
            mg.connect_prolongation(
              create_mg_timer_function(4, "prolongation"));
            mg.connect_edge_prolongation(
              create_mg_timer_function(5, "edge_prolongation"));
            mg.connect_post_smoother_step(
              create_mg_timer_function(6, "post_smoother_step"));
          }

          all_mg_precon_timers.resize(2);

          const auto create_mg_precon_timer_function =
            [&](const unsigned int i) {
              return [i, this](const bool flag) {
                if (flag)
                  all_mg_precon_timers[i].second =
                    std::chrono::system_clock::now();
                else
                  all_mg_precon_timers[i].first +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::system_clock::now() -
                      all_mg_precon_timers[i].second)
                      .count() /
                    1e9;
              };
            };

          preconditioner_mg->connect_transfer_to_mg(
            create_mg_precon_timer_function(0));
          preconditioner_mg->connect_transfer_to_global(
            create_mg_precon_timer_function(1));
        }

      std::string solver_name = "GMRES";

      ReductionControl solver_control(CT::MAX_STEPS_, 1e-14, CT::REDUCE_);
      solver_control.enable_history_data();
      solver_control.log_history(true);

#if SCHWARZTYPE == 0
      SolverGMRES<VectorType> solver(solver_control);
#else
      SolverCG<VectorType> solver(solver_control);
#endif

      Timer              time;
      const unsigned int N         = 5;
      double             best_time = 1e10;

      bool is_converged = true;
      try
        {
          {
            solution[maxlevel] = 0;
            solver.solve(matrix[maxlevel],
                         solution[maxlevel],
                         rhs[maxlevel],
                         *this);
            print_timings();
            clear_timings();
          }

          for (unsigned int i = 0; i < N; ++i)
            {
              time.reset();
              time.start();

              solution[maxlevel] = 0;
              solver.solve(matrix[maxlevel],
                           solution[maxlevel],
                           rhs[maxlevel],
                           *this);

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

      // std::cout << residual_0 << " " << residual_n << std::endl;

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

    // Solve with the FMG cycle and return the reduction rate of a V-cycle
    std::tuple<int, double, double>
    solve_fmg()
    {
      double init_residual = rhs[maxlevel].l2_norm();
      solution[maxlevel]   = 0;

      double res_norm = 0;

      mg_coarse(minlevel, solution[minlevel], rhs[minlevel]);

      for (unsigned int level = minlevel + 1; level <= maxlevel; ++level)
        {
          // set_inhomogeneous_bc<false>(level - 1);

          transfer->prolongate(level, solution[level], solution[level - 1]);

          defect[level] = rhs[level];

          // run v-cycle to obtain correction
          v_cycle(level, true);
        }


      unsigned int it = 0;
      for (; it < CT::MAX_STEPS_; ++it)
        {
          v_cycle(maxlevel, true);

          // set_inhomogeneous_bc<true>(maxlevel);

          matrix[maxlevel].vmult(t[maxlevel], solution[maxlevel]);
          t[maxlevel].sadd(-1., 1., rhs[maxlevel]);
          res_norm = t[maxlevel].l2_norm();

          if (res_norm / init_residual < CT::REDUCE_)
            break;
        }

      return std::make_tuple(it + 1, res_norm, init_residual);
    }

    // Solve with the FMG cycle and return the reduction rate of a V-cycle
    std::tuple<int, double, double>
    solve_vcycle()
    {
      double init_residual = 0;
      double res_norm      = 0;

      init_residual = rhs[maxlevel].l2_norm();

      solution[maxlevel] = 0;

      unsigned int it = 0;
      for (; it < CT::MAX_STEPS_; ++it)
        {
          v_cycle(maxlevel, true);

          // set_inhomogeneous_bc<true>(maxlevel);

          matrix[maxlevel].vmult(t[maxlevel], solution[maxlevel]);
          t[maxlevel].sadd(-1., 1., rhs[maxlevel]);
          res_norm = t[maxlevel].l2_norm();

          if (res_norm / init_residual < CT::REDUCE_)
            break;
        }

      return std::make_tuple(it + 1, res_norm, init_residual);
    }

    // run v-cycle in double precision
    void
    do_fmgcycle()
    {
      mg_coarse(minlevel, solution[minlevel], rhs[minlevel]);

      for (unsigned int level = minlevel + 1; level <= maxlevel; ++level)
        {
          set_inhomogeneous_bc<false>(level - 1);

          transfer->prolongate(level, solution[level], solution[level - 1]);

          defect[level] = rhs[level];

          // run v-cycle to obtain correction
          v_cycle(level, true);
        }
    }

    // run v-cycle in double precision
    void
    do_vcycle()
    {
      v_cycle(maxlevel, true);
    }

    // run smooth in double precision
    void
    do_smooth()
    {
      (mg_smoother).smooth(maxlevel, solution[maxlevel], rhs[maxlevel]);
      cudaDeviceSynchronize();
    }

    // run matrix-vector product in double precision
    void
    do_matvec()
    {
      matrix[maxlevel].vmult(solution[maxlevel], rhs[maxlevel]);
      cudaDeviceSynchronize();
    }

  private:
    // Implement the V-cycle
    void
    v_cycle(const unsigned int level, const bool outer_solution) const
    {
      if (level == minlevel)
        {
          (mg_coarse)(level, solution[level], defect[level]);
          return;
        }

      if (outer_solution == false)
        (mg_smoother).apply(level, solution[level], defect[level]);
      else
        (mg_smoother).smooth(level, solution[level], defect[level]);
      // (mg_smoother).smooth(level, solution[level], defect[level]);

      matrix[level].vmult(t[level], solution[level]);

      t[level].sadd(-1.0, 1.0, defect[level]);

      defect[level - 1] = 0;
      transfer->restrict_and_add(level, defect[level - 1], t[level]);

      v_cycle(level - 1, false);

      transfer->prolongate_and_add(level, solution[level], solution[level - 1]);

      // solution[level] += t[level];

      (mg_smoother).smooth(level, solution[level], defect[level]);
      // (mg_smoother).smooth(level, solution[level], defect[level]);
    }

    template <bool is_zero = false>
    void
    set_inhomogeneous_bc(const unsigned int level)
    {
      unsigned int n_inhomogeneous_bc = inhomogeneous_bc[level].size();
      if (n_inhomogeneous_bc != 0)
        {
          const unsigned int block_size = 256;
          const unsigned int inhomogeneous_n_blocks =
            std::ceil(static_cast<double>(n_inhomogeneous_bc) /
                      static_cast<double>(block_size));
          const unsigned int inhomogeneous_x_n_blocks =
            std::round(std::sqrt(inhomogeneous_n_blocks));
          const unsigned int inhomogeneous_y_n_blocks =
            std::ceil(static_cast<double>(inhomogeneous_n_blocks) /
                      static_cast<double>(inhomogeneous_x_n_blocks));

          dim3 inhomogeneous_grid_dim(inhomogeneous_x_n_blocks,
                                      inhomogeneous_y_n_blocks);
          dim3 inhomogeneous_block_dim(block_size);

          std::vector<unsigned int> inhomogeneous_index_host(
            n_inhomogeneous_bc);
          std::vector<Number> inhomogeneous_value_host(n_inhomogeneous_bc);
          unsigned int        count = 0;
          for (auto &i : inhomogeneous_bc[level])
            {
              inhomogeneous_index_host[count] = i.first;
              inhomogeneous_value_host[count] = i.second;
              count++;
            }

          unsigned int *inhomogeneous_index;
          Number       *inhomogeneous_value;

          cudaError_t cuda_error =
            cudaMalloc(&inhomogeneous_index,
                       n_inhomogeneous_bc * sizeof(unsigned int));
          AssertCuda(cuda_error);

          cuda_error = cudaMemcpy(inhomogeneous_index,
                                  inhomogeneous_index_host.data(),
                                  n_inhomogeneous_bc * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice);
          AssertCuda(cuda_error);

          cuda_error = cudaMalloc(&inhomogeneous_value,
                                  n_inhomogeneous_bc * sizeof(Number));
          AssertCuda(cuda_error);

          cuda_error = cudaMemcpy(inhomogeneous_value,
                                  inhomogeneous_value_host.data(),
                                  n_inhomogeneous_bc * sizeof(Number),
                                  cudaMemcpyHostToDevice);
          AssertCuda(cuda_error);


          set_inhomogeneous_dofs<is_zero, Number>
            <<<inhomogeneous_grid_dim, inhomogeneous_block_dim>>>(
              inhomogeneous_index,
              inhomogeneous_value,
              n_inhomogeneous_bc,
              solution[level].get_values());
          AssertCudaKernel();
        }
    }


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
    mutable MGLevelObject<VectorType> solution;

    /**
     * Original right hand side vector
     */
    mutable MGLevelObject<VectorType> rhs;

    /**
     * Input vector for the cycle. Contains the defect of the
     * outer method projected to the multilevel vectors.
     */
    mutable MGLevelObject<VectorType> defect;

    /**
     * Auxiliary vector.
     */
    mutable MGLevelObject<VectorType> t;


    // MGLevelObject<SmootherType> smooth;

    MGSmootherPrecondition<MatrixType, SmootherType, VectorType> mg_smoother;

    /**
     * The coarse solver
     */
    MGSmootherPrecondition<MatrixType, SmootherTypeCoarse, VectorType>
      mg_smoother_coarse;

    MGCoarseGridApplySmoother<VectorType> mg_coarse;
    // MGCoarseFromSmoother<VectorType, MGLevelObject<SmootherType>> coarse;

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

    mutable std::unique_ptr<
      PreconditionMG<dim, VectorType, MGTransferCUDA<dim, Number>>>
      preconditioner_mg;

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
