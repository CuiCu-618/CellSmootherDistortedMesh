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

#include <functional>

#include "cuda_mg_transfer.cuh"
#include "dealii/cuda_mf.cuh"
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
      oss.setf(std::ios::scientific);

      oss << std::left << std::setw(12) << solver_name << std::setw(4)
          << n_iteration << std::setprecision(4) << std::setw(12) << timing;

      // if (CT::SETS_ == "error_analysis")
      oss << std::left << std::setprecision(4) << std::setw(12) << residual
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
   * @tparam dof_layout
   * @tparam Number full number
   * @tparam smooth_kernel
   * @tparam Number1 vcycle number
   */
  template <int       dim,
            int       fe_degree,
            DoFLayout dof_layout,
            typename Number,
            SmootherVariant smooth_kernel,
            typename Number2>
  class MultigridSolver
  {
  public:
    using VectorType =
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>;
    using VectorType2 =
      LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA>;
    using MatrixType  = LaplaceOperator<dim, fe_degree, Number, dof_layout>;
    using MatrixType2 = LaplaceOperator<dim, fe_degree, Number2, dof_layout>;

    MultigridSolver(
      const DoFHandler<dim>                          &dof_handler,
      const MGLevelObject<MatrixType>                &matrix_dp,
      const MGLevelObject<MatrixType2>               &matrix,
      const MGTransferCUDA<dim, Number, dof_layout>  &mg_transfer_no_boundary,
      const MGTransferCUDA<dim, Number2, dof_layout> &transfer,
      const Function<dim, Number>                    &boundary_values,
      const Function<dim, Number>                    &right_hand_side,
      std::shared_ptr<ConditionalOStream>             pcout,
      const unsigned int                              n_cycles = 1)
      : dof_handler(&dof_handler)
      , matrix_dp(&matrix_dp)
      , matrix(&matrix)
      , transfer(&transfer)
      , mg_transfer_no_boundary(&mg_transfer_no_boundary)
      , minlevel(1)
      , maxlevel(dof_handler.get_triangulation().n_global_levels() - 1)
      , solution(minlevel, maxlevel)
      , rhs(minlevel, maxlevel)
      , residual(minlevel, maxlevel)
      , defect(minlevel, maxlevel)
      , t(minlevel, maxlevel)
      , solution_update(minlevel, maxlevel)
      , smooth(minlevel, maxlevel)
      , coarse(smooth, false)
      , n_cycles(n_cycles)
      , timings(maxlevel + 1)
      , analytic_solution(boundary_values)
      , pcout(pcout)
    {
      AssertDimension(fe_degree, dof_handler.get_fe().degree);

      for (unsigned int level = minlevel; level <= maxlevel; ++level)
        {
          matrix[level].initialize_dof_vector(defect[level]);
          t[level].reinit(defect[level]);
          solution_update[level].reinit(defect[level]);

          matrix_dp[level].initialize_dof_vector(solution[level]);
          rhs[level].reinit(solution[level]);
          residual[level].reinit(solution[level]);
        }

      // Timer time;
      cudaEvent_t start, stop;
      AssertCuda(cudaEventCreate(&start));
      AssertCuda(cudaEventCreate(&stop));

      // set up a mapping for the geometry representation
      MappingQ1<dim> mapping;

      // interpolate the inhomogeneous boundary conditions
      inhomogeneous_bc.clear();
      inhomogeneous_bc.resize(maxlevel + 1);
      if (CT::SETS_ == "error_analysis")
        for (unsigned int level = minlevel; level <= maxlevel; ++level)
          {
            Quadrature<dim - 1> face_quad(
              dof_handler.get_fe().get_unit_face_support_points());
            FEFaceValues<dim>                    fe_values(mapping,
                                        dof_handler.get_fe(),
                                        face_quad,
                                        update_quadrature_points);
            std::vector<types::global_dof_index> face_dof_indices(
              dof_handler.get_fe().dofs_per_face);

            typename DoFHandler<dim>::cell_iterator cell =
                                                      dof_handler.begin(level),
                                                    endc =
                                                      dof_handler.end(level);
            for (; cell != endc; ++cell)
              if (cell->level_subdomain_id() !=
                  numbers::artificial_subdomain_id)
                for (unsigned int face_no = 0;
                     face_no < GeometryInfo<dim>::faces_per_cell;
                     ++face_no)
                  if (cell->at_boundary(face_no))
                    {
                      const typename DoFHandler<dim>::face_iterator face =
                        cell->face(face_no);
                      face->get_mg_dof_indices(level, face_dof_indices);
                      fe_values.reinit(cell, face_no);
                      for (unsigned int i = 0; i < face_dof_indices.size(); ++i)
                        if (dof_handler.locally_owned_mg_dofs(level).is_element(
                              face_dof_indices[i]))
                          {
                            const double value = analytic_solution.value(
                              fe_values.quadrature_point(i));
                            if (value != 0.0)
                              inhomogeneous_bc[level][face_dof_indices[i]] =
                                value;
                          }
                    }
          }

      for (int level = maxlevel; level >= minlevel; --level)
        {
          // evaluate the right hand side in the equation, including the
          // residual from the inhomogeneous boundary conditions
          set_inhomogeneous_bc<false>(level);
          rhs[level] = 0.;
          if (level == maxlevel)
            if (CT::SETS_ == "error_analysis")
              matrix_dp[level].compute_residual(rhs[level],
                                                solution[level],
                                                right_hand_side,
                                                level);
            else
              rhs[level] = 1.;
          else
            mg_transfer_no_boundary.restrict_and_add(level + 1,
                                                     rhs[level],
                                                     rhs[level + 1]);
        }
      float gpu_time = 0.0f;

      cudaEventRecord(start);
      {
        typename SmootherType::AdditionalData additional_data;
        additional_data.relaxation = 1.;
        additional_data.patch_per_block =
          fe_degree == 1 ? 16 : (fe_degree == 2 ? 2 : 1);
        additional_data.granularity_scheme = CT::GRANULARITY_;

        for (unsigned int level = minlevel; level <= maxlevel; ++level)
          smooth[level].initialize(matrix[level], additional_data);
      }
      cudaEventRecord(stop);
      cudaDeviceSynchronize();
      AssertCuda(cudaEventElapsedTime(&gpu_time, start, stop));

      *pcout << "Smoother setup time:    " << gpu_time / 1e3 << "s"
             << std::endl;

      AssertCuda(cudaEventDestroy(start));
      AssertCuda(cudaEventDestroy(stop));
    }

    // Print a summary of computation times on the various levels
    void
    print_wall_times(unsigned int N)
    {
      *pcout << "Coarse solver " << (int)timings[minlevel][1] / N
             << " times: " << timings[minlevel][0] / 1e3 / N << " tot prec "
             << timings[minlevel][2] / 1e3 / N << std::endl;
      *pcout
        << "level  smoother    mg_mv     mg_vec    restrict  prolongate  inhomBC   total"
        << std::endl;
      double l_L = 0;
      for (unsigned int level = minlevel + 1; level <= maxlevel; ++level)
        {
          *pcout << "L" << std::setw(2) << std::left << level << "    ";
          *pcout << std::setprecision(4) << std::setw(12)
                 << timings[level][5] / 1e3 / N << std::setw(10)
                 << timings[level][0] / 1e3 / N << std::setw(10)
                 << timings[level][4] / 1e3 / N << std::setw(10)
                 << timings[level][1] / 1e3 / N << std::setw(12)
                 << timings[level][2] / 1e3 / N << std::setw(10)
                 << timings[level][3] / 1e3 / N << std::setw(10)
                 << (timings[level][5] + timings[level][0] + timings[level][4] +
                     timings[level][1] + timings[level][2] +
                     timings[level][3]) /
                      1e3 / N
                 << std::endl;
          if (level < maxlevel)
            {
              l_L += timings[level][5] / 1e3 / N;
              l_L += timings[level][0] / 1e3 / N;
              l_L += timings[level][4] / 1e3 / N;
              l_L += timings[level][1] / 1e3 / N;
              l_L += timings[level][2] / 1e3 / N;
              l_L += timings[level][3] / 1e3 / N;
            }
        }
      *pcout << "l < L: " << l_L << "\t grid transfer:"
             << (timings[maxlevel][1] + timings[maxlevel][2]) / 1e3 / N
             << std::endl;

      *pcout << std::setprecision(5);

      for (unsigned int l = 0; l < timings.size(); ++l)
        for (unsigned int j = 0; j < timings[l].size(); ++j)
          timings[l][j] = 0.;
    }



    // Return the solution vector for further processing
    const VectorType &
    get_solution()
    {
      set_inhomogeneous_bc<false>(maxlevel);
      return solution[maxlevel];
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

      for (unsigned int s = 0; s < 4; ++s)
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

      {
        for (unsigned int l = 0; l < timings.size(); ++l)
          for (unsigned int j = 0; j < timings[l].size(); ++j)
            timings[l][j] = 0.;
      }

      return comp_data;
    }


    std::vector<SolverData>
    solve()
    {
      *pcout << "Solving...\n";

      std::string solver_name = "";

      auto solver = [&](auto kernel) {
        Timer              time;
        const unsigned int N         = 10;
        double             best_time = 1e10;
        for (unsigned int i = 0; i < N; ++i)
          {
            time.reset();
            time.start();
            kernel(this);
            best_time = std::min(time.wall_time(), best_time);
          }

        print_wall_times(N);

        const auto [n_iter, residual_n, residual_0] = kernel(this);

        // *** average reduction: r_n = rho^n * r_0
        const double rho =
          std::pow(residual_n / residual_0, static_cast<double>(1. / n_iter));
        const double convergence_rate =
          1. / n_iter * std::log10(residual_0 / residual_n);

        size_t free_mem, total_mem;
        AssertCuda(cudaMemGetInfo(&free_mem, &total_mem));

        int mem_usage = (total_mem - free_mem) / 1024 / 1024;

        SolverData data;
        data.solver_name      = solver_name;
        data.n_iteration      = n_iter;
        data.residual         = residual_n;
        data.reduction_rate   = rho;
        data.convergence_rate = convergence_rate;
        data.timing           = best_time;
        data.mem_usage        = mem_usage;
        solver_data.push_back(data);
      };

      for (unsigned int s = 0; s < 1; ++s)
        {
          switch (s)
            {
              case 0:
                {
                  auto kernel = std::mem_fn(&MultigridSolver::solve_gmres);
                  solver_name = "GMRES";
                  solver(kernel);
                  break;
                }
              default:
                AssertThrow(false, ExcMessage("Invalid Solver Variant."));
            }
        }

      return solver_data;
    }

    /**
     * compute the average reduction rho over n iterations and the
     * fractional number of iterations to achieve the requested
     * reduction (relative stopping criterion)
     */
    std::pair<double, double>
    compute_fractional_steps(const ReductionControl &solver_control)
    {
      const double residual_0 = solver_control.initial_value();
      const double residual_n = solver_control.last_value();
      const int    n = solver_control.last_step(); // number of iterations
      const double reduction = solver_control.reduction(); // relative tolerance

      // *** average reduction: r_n = rho^n * r_0
      const double rho =
        std::pow(residual_n / residual_0, static_cast<double>(1. / n));

      /**
       * since r_n <= reduction * r_0 we can compute the fractional
       * number of iterations n_frac that is sufficient to achieve the
       * desired reduction:
       *    rho^n_frac = reduction   <=>   n_frac = log(reduction)/log(rho)
       */
      const double n_frac = std::log(reduction) / std::log(rho);

      return std::make_pair(n_frac, rho);
    }

    // Solve with the conjugate gradient method preconditioned by the V-cycle
    // (invoking this->vmult) and return the number of iterations and the
    // reduction rate per CG iteration
    std::tuple<int, double, double>
    solve_gmres()
    {
      ReductionControl solver_control(CT::MAX_STEPS_, 1e-15, CT::REDUCE_);

      SolverGMRES<VectorType> solver(solver_control);
      solution[maxlevel] = 0;
      solver.solve((*matrix_dp)[maxlevel],
                   solution[maxlevel],
                   rhs[maxlevel],
                   *this);

      return std::make_tuple(solver_control.last_step(),
                             solver_control.last_value(),
                             solver_control.initial_value());
    }

    // Implement the vmult() function needed by the preconditioner interface
    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      // Timer time1, time;
      cudaEvent_t start, stop;
      AssertCuda(cudaEventCreate(&start));
      AssertCuda(cudaEventCreate(&stop));
      float gpu_time1 = 0.0f, gpu_time2 = 0.0f;

      cudaEventRecord(start);
      // defect[maxlevel] = src;
      convert_precision<true>(defect[maxlevel], src);

      cudaEventRecord(stop);
      cudaDeviceSynchronize();
      AssertCuda(cudaEventElapsedTime(&gpu_time1, start, stop));
      timings[maxlevel][4] += gpu_time1;

      cudaEventRecord(start);
      v_cycle(maxlevel, 1);

      cudaEventRecord(stop);
      cudaDeviceSynchronize();
      AssertCuda(cudaEventElapsedTime(&gpu_time2, start, stop));

      timings[minlevel][2] += gpu_time1 + gpu_time2;

      cudaEventRecord(start);
      // dst = solution_update[maxlevel];
      convert_precision<false>(dst, solution_update[maxlevel]);

      cudaEventRecord(stop);
      cudaDeviceSynchronize();
      AssertCuda(cudaEventElapsedTime(&gpu_time1, start, stop));
      timings[maxlevel][4] += gpu_time1;
      timings[minlevel][2] += gpu_time1;

      AssertCuda(cudaEventDestroy(start));
      AssertCuda(cudaEventDestroy(stop));
    }

    // run matrix-vector product in double precision
    void
    do_matvec()
    {
      (*matrix_dp)[maxlevel].vmult(residual[maxlevel], solution[maxlevel]);
      cudaDeviceSynchronize();
    }

    // run matrix-vector product in single precision
    void
    do_matvec_smoother()
    {
      (*matrix)[maxlevel].vmult(solution_update[maxlevel], defect[maxlevel]);
      cudaDeviceSynchronize();
    }

    // run smoother in single precision
    void
    do_smoother()
    {
      (smooth)[maxlevel].step(solution_update[maxlevel], defect[maxlevel]);
      cudaDeviceSynchronize();
    }

    // run v-cycle in single precision
    void
    do_vcycle()
    {
      v_cycle(maxlevel, true);
    }

    double
    memory_consumption_GMRES()
    {
      double result = sizeof(*this);
      result += solution.memory_consumption();
      result += rhs.memory_consumption();
      result += transfer.memory_consumption();
      result += defect.memory_consumption();
      result += t.memory_consumption();
      result += solution_update.memory_consumption();
      result += matrix.memory_consumption();
      result += matrix_dp[maxlevel].memory_consumption();
      result += smoother_mem;

      double scale = 1024 * 1024;
      result       = result / scale;


      *pcout << "GMRES GPU Memory Usage [MB]" << std::endl
             << "System matrix: "
             << matrix_dp[maxlevel].memory_consumption() / scale << std::endl
             << "Level matrix:  " << matrix.memory_consumption() / scale
             << std::endl
             << "Solution:      " << solution.memory_consumption() / scale
             << std::endl
             << "System rhs:    " << rhs.memory_consumption() / scale
             << std::endl
             << "MG transfer:   " << transfer.memory_consumption() / scale
             << std::endl
             << "MG smoother:   " << smoother_mem / scale << std::endl
             << "MG Auxiliary:  "
             << (defect.memory_consumption() +
                 solution_update.memory_consumption() +
                 t.memory_consumption()) /
                  scale
             << std::endl
             << "Total:         " << result << "\n\n";


      return result;
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
          (coarse)(level, solution_update[level], defect[level]);
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
            (smooth)[level].vmult(solution_update[level], defect[level]);
          else
            (smooth)[level].step(solution_update[level], defect[level]);
          cudaEventRecord(stop);
          cudaDeviceSynchronize();
          AssertCuda(cudaEventElapsedTime(&gpu_time, start, stop));
          timings[level][5] += gpu_time;

          cudaEventRecord(start);
          (*matrix)[level].vmult(t[level], solution_update[level]);
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
                                       t[level],
                                       solution_update[level - 1]);
          cudaEventRecord(stop);
          cudaDeviceSynchronize();
          AssertCuda(cudaEventElapsedTime(&gpu_time, start, stop));
          timings[level][2] += gpu_time;

          cudaEventRecord(start);
          solution_update[level] += t[level];
          cudaEventRecord(stop);
          cudaDeviceSynchronize();
          AssertCuda(cudaEventElapsedTime(&gpu_time, start, stop));
          timings[level][4] += gpu_time;

          cudaEventRecord(start);
          (smooth)[level].step(solution_update[level], defect[level]);
          cudaEventRecord(stop);
          cudaDeviceSynchronize();
          AssertCuda(cudaEventElapsedTime(&gpu_time, start, stop));
          timings[level][5] += gpu_time;
        }
      AssertCuda(cudaEventDestroy(start));
      AssertCuda(cudaEventDestroy(stop));
    }

    const SmartPointer<const DoFHandler<dim>> dof_handler;

    const SmartPointer<const MGLevelObject<MatrixType>>  matrix_dp;
    const SmartPointer<const MGLevelObject<MatrixType2>> matrix;

    const SmartPointer<const MGTransferCUDA<dim, Number, dof_layout>>
      mg_transfer_no_boundary;
    const SmartPointer<const MGTransferCUDA<dim, Number2, dof_layout>> transfer;

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
     * Residual vector before it is passed down into float through the v-cycle
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

    /**
     * The smoother object
     */
    using SmootherType =
      PatchSmoother<MatrixType2, dim, fe_degree, smooth_kernel, dof_layout>;

    MGLevelObject<SmootherType> smooth;

    /**
     * The coarse solver
     */
    MGCoarseFromSmoother<VectorType2, MGLevelObject<SmootherType>> coarse;

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
  };



  template <int       dim,
            int       fe_degree,
            DoFLayout dof_layout,
            typename Number,
            SmootherVariant smooth_kernel>
  class MultigridSolver<dim,
                        fe_degree,
                        dof_layout,
                        Number,
                        smooth_kernel,
                        Number>
  {
  public:
    using VectorType =
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>;
    using MatrixType = LaplaceOperator<dim, fe_degree, Number, dof_layout>;

    MultigridSolver(const DoFHandler<dim>           &dof_handler,
                    const MGLevelObject<MatrixType> &matrix_dp,
                    const MGLevelObject<MatrixType> &,
                    const MGTransferCUDA<dim, Number, dof_layout> &transfer_dp,
                    const MGTransferCUDA<dim, Number, dof_layout> &,
                    const Function<dim, Number>        &boundary_values,
                    const Function<dim, Number>        &right_hand_side,
                    std::shared_ptr<ConditionalOStream> pcout,
                    const unsigned int                  n_cycles = 1)
      : dof_handler(&dof_handler)
      , matrix(&matrix_dp)
      , transfer(&transfer_dp)
      , minlevel(1)
      , maxlevel(dof_handler.get_triangulation().n_global_levels() - 1)
      , solution(minlevel, maxlevel)
      , rhs(minlevel, maxlevel)
      , defect(minlevel, maxlevel)
      , t(minlevel, maxlevel)
      , smooth(minlevel, maxlevel)
      , coarse(smooth, false)
      , n_cycles(n_cycles)
      , analytic_solution(boundary_values)
      , pcout(pcout)
    {
      AssertDimension(fe_degree, dof_handler.get_fe().degree);

      for (unsigned int level = minlevel; level <= maxlevel; ++level)
        {
          matrix_dp[level].initialize_dof_vector(solution[level]);
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
            Quadrature<dim - 1> face_quad(
              dof_handler.get_fe().get_unit_face_support_points());
            FEFaceValues<dim>                    fe_values(mapping,
                                        dof_handler.get_fe(),
                                        face_quad,
                                        update_quadrature_points);
            std::vector<types::global_dof_index> face_dof_indices(
              dof_handler.get_fe().dofs_per_face);

            typename DoFHandler<dim>::cell_iterator cell =
                                                      dof_handler.begin(level),
                                                    endc =
                                                      dof_handler.end(level);
            for (; cell != endc; ++cell)
              if (cell->level_subdomain_id() !=
                  numbers::artificial_subdomain_id)
                for (unsigned int face_no = 0;
                     face_no < GeometryInfo<dim>::faces_per_cell;
                     ++face_no)
                  if (cell->at_boundary(face_no))
                    {
                      const typename DoFHandler<dim>::face_iterator face =
                        cell->face(face_no);
                      face->get_mg_dof_indices(level, face_dof_indices);
                      fe_values.reinit(cell, face_no);
                      for (unsigned int i = 0; i < face_dof_indices.size(); ++i)
                        if (dof_handler.locally_owned_mg_dofs(level).is_element(
                              face_dof_indices[i]))
                          {
                            const double value = analytic_solution.value(
                              fe_values.quadrature_point(i));
                            if (value != 0.0)
                              inhomogeneous_bc[level][face_dof_indices[i]] =
                                value;
                          }
                    }
          }

      for (int level = maxlevel; level >= minlevel; --level)
        {
          // evaluate the right hand side in the equation, including the
          // residual from the inhomogeneous boundary conditions
          set_inhomogeneous_bc<false>(level);
          rhs[level] = 0.;
          if (level == maxlevel)
            if (CT::SETS_ == "error_analysis")
              matrix_dp[level].compute_residual(rhs[level],
                                                solution[level],
                                                right_hand_side,
                                                level);
            else
              rhs[level] = 1.;
          else
            transfer_dp.restrict_and_add(level + 1, rhs[level], rhs[level + 1]);
        }

      {
        Timer time;

        typename SmootherType::AdditionalData additional_data;
        additional_data.relaxation = 1.;
        additional_data.patch_per_block =
          fe_degree == 1 ? 16 : (fe_degree == 2 ? 2 : 1);
        additional_data.granularity_scheme = CT::GRANULARITY_;

        for (unsigned int level = minlevel; level <= maxlevel; ++level)
          smooth[level].initialize(matrix_dp[level], additional_data);

        *pcout << "Smoother setup time:    " << time.wall_time() << "s"
               << std::endl;
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

      for (unsigned int s = 0; s < 4; ++s)
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

    std::vector<SolverData>
    solve()
    {
      *pcout << "Solving...\n";

      std::string solver_name = "";

      auto solver = [&](auto kernel) {
        Timer              time;
        const unsigned int N         = 10;
        double             best_time = 1e10;
        for (unsigned int i = 0; i < N; ++i)
          {
            time.reset();
            time.start();
            kernel(this);
            best_time = std::min(time.wall_time(), best_time);
          }

        const auto [n_iter, residual_n, residual_0] = kernel(this);

        // *** average reduction: r_n = rho^n * r_0
        const double rho =
          std::pow(residual_n / residual_0, static_cast<double>(1. / n_iter));
        const double convergence_rate =
          1. / n_iter * std::log10(residual_0 / residual_n);

        size_t free_mem, total_mem;
        AssertCuda(cudaMemGetInfo(&free_mem, &total_mem));

        int mem_usage = (total_mem - free_mem) / 1024 / 1024;

        SolverData data;
        data.solver_name      = solver_name;
        data.n_iteration      = n_iter;
        data.residual         = residual_n;
        data.reduction_rate   = rho;
        data.convergence_rate = convergence_rate;
        data.timing           = best_time;
        data.mem_usage        = mem_usage;
        solver_data.push_back(data);
      };

      for (unsigned int s = 0; s < 3; ++s)
        {
          switch (s)
            {
              case 0:
                {
                  auto kernel = std::mem_fn(&MultigridSolver::solve_fmg);
                  solver_name = "Linear-FMG";
                  solver(kernel);
                  break;
                }
              case 1:
                {
                  auto kernel = std::mem_fn(&MultigridSolver::solve_vcycle);
                  solver_name = "V-cycles";
                  solver(kernel);
                  break;
                }
              case 2:
                {
                  auto kernel = std::mem_fn(&MultigridSolver::solve_gmres);
                  solver_name = "GMRES";
                  solver(kernel);
                  break;
                }
              default:
                AssertThrow(false, ExcMessage("Invalid Solver Variant."));
            }
        }

      return solver_data;
    }

    // Implement the vmult() function needed by the preconditioner interface
    void
    vmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
          const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
            &src) const
    {
      defect[maxlevel] = src;
      v_cycle(maxlevel, 1);
      dst = solution[maxlevel];
    }

    // Solve with the conjugate gradient method preconditioned by the V-cycle
    // (invoking this->vmult) and return the number of iterations and the
    // reduction rate per GMRES iteration
    std::tuple<int, double, double>
    solve_gmres()
    {
      ReductionControl solver_control(CT::MAX_STEPS_, 1e-15, CT::REDUCE_);

      SolverGMRES<VectorType> solver(solver_control);
      solution[maxlevel] = 0;
      solver.solve((*matrix)[maxlevel],
                   solution[maxlevel],
                   rhs[maxlevel],
                   *this);

      return std::make_tuple(solver_control.last_step(),
                             solver_control.last_value(),
                             solver_control.initial_value());
    }

    // Solve with the FMG cycle and return the reduction rate of a V-cycle
    std::tuple<int, double, double>
    solve_fmg()
    {
      double init_residual = rhs[maxlevel].l2_norm();
      solution[maxlevel]   = 0;

      double res_norm = 0;

      coarse(minlevel, solution[minlevel], rhs[minlevel]);

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

          set_inhomogeneous_bc<true>(maxlevel);

          (*matrix)[maxlevel].vmult(t[maxlevel], solution[maxlevel]);
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

          set_inhomogeneous_bc<true>(maxlevel);

          (*matrix)[maxlevel].vmult(t[maxlevel], solution[maxlevel]);
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
      coarse(minlevel, solution[minlevel], rhs[minlevel]);

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
      (smooth)[maxlevel].step(solution[maxlevel], rhs[maxlevel]);
      cudaDeviceSynchronize();
    }

    // run matrix-vector product in double precision
    void
    do_matvec()
    {
      (*matrix)[maxlevel].vmult(solution[maxlevel], rhs[maxlevel]);
      cudaDeviceSynchronize();
    }

  private:
    // Implement the V-cycle
    void
    v_cycle(const unsigned int level, const bool outer_solution) const
    {
      if (level == minlevel)
        {
          (coarse)(level, solution[level], defect[level]);
          return;
        }

      if (outer_solution == false)
        (smooth)[level].vmult(solution[level], defect[level]);
      else
        (smooth)[level].step(solution[level], defect[level]);

      (*matrix)[level].vmult(t[level], solution[level]);

      t[level].sadd(-1.0, 1.0, defect[level]);

      defect[level - 1] = 0;
      transfer->restrict_and_add(level, defect[level - 1], t[level]);

      v_cycle(level - 1, false);

      transfer->prolongate_and_add(level, t[level], solution[level - 1]);

      solution[level] += t[level];

      (smooth)[level].step(solution[level], defect[level]);
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


    const SmartPointer<const DoFHandler<dim>>           dof_handler;
    const SmartPointer<const MGLevelObject<MatrixType>> matrix;
    const SmartPointer<const MGTransferCUDA<dim, Number, dof_layout>> transfer;

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

    /**
     * The smoother object
     */
    using SmootherType =
      PatchSmoother<MatrixType, dim, fe_degree, smooth_kernel, dof_layout>;

    MGLevelObject<SmootherType> smooth;

    /**
     * The coarse solver
     */
    MGCoarseFromSmoother<VectorType, MGLevelObject<SmootherType>> coarse;

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

} // namespace PSMF

#endif // SOLVER_CUH
