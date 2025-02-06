/**
 * @file poisson.cu
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief Discontinuous Galerkin methods for poisson problems.
 * @version 1.0
 * @date 2023-02-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/cuda.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <helper_cuda.h>

#include <fstream>

#include "app_utilities.h"
#include "ct_parameter.h"
#include "patch_base.cuh"
#include "solver.cuh"
#include "utilities.cuh"


namespace Step64
{
  using namespace dealii;

  template <int dim, typename Number>
  class Solution : public Function<dim, Number>
  {
    static constexpr auto PI = numbers::PI;

  public:
    Solution()
      : Function<dim, Number>()
    {}

    virtual Number
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      Number return_value = 0.0;

      if (dim == 2)
        {
          return_value = std::sin(PI * p[0]) * std::sin(PI * p[1]);
        }
      else if (dim == 3)
        {
          return_value =
            std::sin(PI * p[0]) * std::sin(PI * p[1]) * std::sin(PI * p[2]);
        }
      else
        Assert(false, ExcNotImplemented());

      return return_value;
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> return_gradient;

      if (dim == 2)
        {
          return_gradient[0] = PI * std::cos(PI * p[0]) * std::sin(PI * p[1]);
          return_gradient[1] = PI * std::cos(PI * p[1]) * std::sin(PI * p[0]);
        }
      else if (dim == 3)
        {
          return_gradient[0] = PI * std::cos(PI * p[0]) * std::sin(PI * p[1]) *
                               std::sin(PI * p[2]);
          return_gradient[1] = PI * std::cos(PI * p[1]) * std::sin(PI * p[0]) *
                               std::sin(PI * p[2]);
          return_gradient[2] = PI * std::cos(PI * p[2]) * std::sin(PI * p[0]) *
                               std::sin(PI * p[1]);
        }
      else
        Assert(false, ExcNotImplemented());

      return return_gradient;
    }

    virtual SymmetricTensor<2, dim>
    hessian(const Point<dim> &p,
            const unsigned int /*component*/ = 0) const override
    {
      SymmetricTensor<2, dim> return_hessian;

      if (dim == 2)
        {
          return_hessian[0][0] =
            -PI * PI * std::sin(PI * p[0]) * std::sin(PI * p[1]);
          return_hessian[0][1] =
            PI * PI * std::cos(PI * p[0]) * std::cos(PI * p[1]);
          return_hessian[1][1] =
            -PI * PI * std::sin(PI * p[0]) * std::sin(PI * p[1]);
        }
      else if (dim == 3)
        {
          return_hessian[0][0] = -PI * PI * std::sin(PI * p[0]) *
                                 std::sin(PI * p[1]) * std::sin(PI * p[2]);
          return_hessian[0][1] = PI * PI * std::cos(PI * p[0]) *
                                 std::cos(PI * p[1]) * std::sin(PI * p[2]);
          return_hessian[0][2] = PI * PI * std::cos(PI * p[0]) *
                                 std::cos(PI * p[2]) * std::sin(PI * p[1]);
          return_hessian[1][1] = -PI * PI * std::sin(PI * p[0]) *
                                 std::sin(PI * p[1]) * std::sin(PI * p[2]);
          return_hessian[1][2] = PI * PI * std::cos(PI * p[1]) *
                                 std::cos(PI * p[2]) * std::sin(PI * p[0]);
          return_hessian[2][2] = -PI * PI * std::sin(PI * p[0]) *
                                 std::sin(PI * p[1]) * std::sin(PI * p[2]);
        }
      else
        Assert(false, ExcNotImplemented());

      return return_hessian;
    }
  };

  template <int dim, typename Number>
  class RightHandSide : public Function<dim, Number>
  {
    static constexpr auto PI = numbers::PI;

  public:
    RightHandSide()
      : Function<dim, Number>()
    {}

    virtual Number
    value(const Point<dim> &p, const unsigned int = 0) const override final
    {
      Number return_value = 0.0;

      if (dim == 2)
        {
          return_value =
            4 * std::pow(PI, 4.0) * std::sin(PI * p[0]) * std::sin(PI * p[1]);
        }
      else if (dim == 3)
        {
          return_value = 9 * std::pow(PI, 4.0) * std::sin(PI * p[0]) *
                         std::sin(PI * p[1]) * std::sin(PI * p[2]);
        }
      else
        Assert(false, ExcNotImplemented());

      return return_value;
    }
  };

  template <int dim, int fe_degree>
  class LaplaceProblem
  {
  public:
    using full_number   = double;
    using vcycle_number = CT::VCYCLE_NUMBER_;
    using MatrixFreeDP  = PSMF::LevelVertexPatch<dim, fe_degree, full_number>;
    using MatrixFreeSP  = PSMF::LevelVertexPatch<dim, fe_degree, vcycle_number>;

    LaplaceProblem();
    ~LaplaceProblem();
    void
    run(const unsigned int n_cycles);

  private:
    void
    setup_system();
    void
    assemble_mg();
    void
    solve_mg(unsigned int n_mg_cycles);
    std::pair<double, double>
    compute_error();

    template <PSMF::LocalSolverVariant local_solver,
              PSMF::LaplaceVariant     laplace,
              PSMF::LaplaceVariant     smooth_vmult,
              PSMF::SmootherVariant    smooth_inv>
    void
    do_solve(unsigned int k, unsigned int call_count);

    Triangulation<dim>                  triangulation;
    std::shared_ptr<FiniteElement<dim>> fe;
    DoFHandler<dim>                     dof_handler;
    MappingQ1<dim>                      mapping;
    double                              setup_time;

    std::vector<ConvergenceTable> info_table;

    std::fstream                        fout;
    std::shared_ptr<ConditionalOStream> pcout;

    MGLevelObject<std::shared_ptr<MatrixFreeDP>> mfdata_dp;
    MGLevelObject<std::shared_ptr<MatrixFreeSP>> mfdata_sp;
    MGConstrainedDoFs                            mg_constrained_dofs;
    AffineConstraints<double>                    constraints;

    LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
      ghost_solution_host;

    PSMF::MGTransferCUDA<dim, vcycle_number> transfer;
  };

  template <int dim, int fe_degree>
  LaplaceProblem<dim, fe_degree>::LaplaceProblem()
    : triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
    , fe([&]() -> std::shared_ptr<FiniteElement<dim>> {
      if (CT::DOF_LAYOUT_ == PSMF::DoFLayout::Q)
        return std::make_shared<FE_Q<dim>>(fe_degree);
      else if (CT::DOF_LAYOUT_ == PSMF::DoFLayout::DGQ)
        return std::make_shared<FE_DGQHermite<dim>>(fe_degree);
      return std::shared_ptr<FiniteElement<dim>>();
    }())
    , dof_handler(triangulation)
    , setup_time(0.)
    , pcout(std::make_shared<ConditionalOStream>(std::cout, false))
  {
    const auto filename = Util::get_filename();
    fout.open(filename + ".log", std::ios_base::out);
    pcout = std::make_shared<ConditionalOStream>(fout, true);

    auto tmp = std::max(CT::LAPLACE_TYPE_.size(), CT::LOCAL_SOLVER_.size());
    info_table.resize(std::max(tmp, CT::SMOOTH_INV_.size()));
  }

  template <int dim, int fe_degree>
  LaplaceProblem<dim, fe_degree>::~LaplaceProblem()
  {
    fout.close();
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::setup_system()
  {
    Timer time;
    setup_time = 0;

    dof_handler.distribute_dofs(*fe);
    dof_handler.distribute_mg_dofs();
    const unsigned int nlevels = triangulation.n_global_levels();
    for (unsigned int level = 0; level < nlevels; ++level)
      Util::Lexicographic(dof_handler, level);
    Util::Lexicographic(dof_handler);

    *pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << " = ("
           << ((int)std::pow(dof_handler.n_dofs() * 1.0000001, 1. / dim) - 1) /
                fe->degree
           << " x " << fe->degree << " + 1)^" << dim << std::endl;

    constraints.clear();
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Solution<dim, full_number>(),
                                             constraints);
    constraints.close();

    setup_time += time.wall_time();

    *pcout << "DoF setup time:         " << setup_time << "s" << std::endl;
  }
  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::assemble_mg()
  {
    // Initialization of Dirichlet boundaries
    std::set<types::boundary_id> dirichlet_boundary;
    dirichlet_boundary.insert(0);
    mg_constrained_dofs.initialize(dof_handler);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                       dirichlet_boundary);

    // set up a mapping for the geometry representation
    MappingQ1<dim> mapping;

    unsigned int minlevel = 1;
    unsigned int maxlevel = triangulation.n_global_levels() - 1;

    mfdata_dp.resize(1, maxlevel);

    if (std::is_same_v<vcycle_number, float>)
      mfdata_sp.resize(1, maxlevel);

    Timer time;
    for (unsigned int level = minlevel; level <= maxlevel; ++level)
      {
        // IndexSet relevant_dofs;
        // DoFTools::extract_locally_relevant_level_dofs(dof_handler,
        //                                               level,
        //                                               relevant_dofs);
        // double-precision matrix-free data
        {
          // AffineConstraints<full_number> level_constraints;
          // level_constraints.reinit(relevant_dofs);
          // level_constraints.add_lines(
          //   mg_constrained_dofs.get_boundary_indices(level));
          // level_constraints.close();

          typename MatrixFreeDP::AdditionalData additional_data;
          additional_data.relaxation         = CT::DAMPING_;
          additional_data.use_coloring       = false;
          additional_data.patch_per_block    = CT::PATCH_PER_BLOCK_;
          additional_data.granularity_scheme = CT::GRANULARITY_;

          mfdata_dp[level] = std::make_shared<MatrixFreeDP>();
          mfdata_dp[level]->reinit(dof_handler,
                                   mg_constrained_dofs,
                                   level,
                                   additional_data);
        }

        // single-precision matrix-free data
        if (std::is_same_v<vcycle_number, float>)
          {
            // AffineConstraints<vcycle_number> level_constraints;
            // level_constraints.reinit(relevant_dofs);
            // level_constraints.add_lines(
            //   mg_constrained_dofs.get_boundary_indices(level));
            // level_constraints.close();

            typename MatrixFreeSP::AdditionalData additional_data;
            additional_data.relaxation         = CT::DAMPING_;
            additional_data.use_coloring       = false;
            additional_data.patch_per_block    = CT::PATCH_PER_BLOCK_;
            additional_data.granularity_scheme = CT::GRANULARITY_;

            mfdata_sp[level] = std::make_shared<MatrixFreeSP>();
            mfdata_sp[level]->reinit(dof_handler,
                                     mg_constrained_dofs,
                                     level,
                                     additional_data);
          }
      }

    *pcout << "Matrix-free setup time: " << time.wall_time() << "s"
           << std::endl;

    time.restart();

    transfer.initialize_constraints(mg_constrained_dofs);
    transfer.build(dof_handler);

    *pcout << "MG transfer setup time: " << time.wall_time() << "s"
           << std::endl;
  }

  template <int dim, int fe_degree>
  template <PSMF::LocalSolverVariant local_solver,
            PSMF::LaplaceVariant     laplace,
            PSMF::LaplaceVariant     smooth_vmult,
            PSMF::SmootherVariant    smooth_inv>
  void
  LaplaceProblem<dim, fe_degree>::do_solve(unsigned int k,
                                           unsigned int call_count)
  {
    PSMF::MultigridSolver<dim,
                          fe_degree,
                          full_number,
                          local_solver,
                          laplace,
                          smooth_vmult,
                          smooth_inv,
                          vcycle_number>
      solver(dof_handler,
             mfdata_dp,
             mfdata_sp,
             transfer,
             Solution<dim, full_number>(),
             RightHandSide<dim, full_number>(),
             pcout,
             1);
    if (CT::LAPLACE_TYPE_.size() > 1)
      *pcout << "\nMG with [" << LaplaceToString(CT::LAPLACE_TYPE_[k]) << " "
             << LaplaceToString(CT::SMOOTH_VMULT_[0]) << " "
             << SmootherToString(CT::SMOOTH_INV_[0]) << " "
             << LocalSolverToString(CT::LOCAL_SOLVER_[0]) << "]\n";
    else if (CT::SMOOTH_INV_.size() > 1)
      *pcout << "\nMG with [" << LaplaceToString(CT::LAPLACE_TYPE_[0]) << " "
             << LaplaceToString(CT::SMOOTH_VMULT_[0]) << " "
             << SmootherToString(CT::SMOOTH_INV_[k]) << " "
             << LocalSolverToString(CT::LOCAL_SOLVER_[0]) << "]\n";
    else if (CT::LOCAL_SOLVER_.size() > 1)
      *pcout << "\nMG with [" << LaplaceToString(CT::LAPLACE_TYPE_[0]) << " "
             << LaplaceToString(CT::SMOOTH_VMULT_[0]) << " "
             << SmootherToString(CT::SMOOTH_INV_[0]) << " "
             << LocalSolverToString(CT::LOCAL_SOLVER_[k]) << "]\n";
    else
      *pcout << "\nMG with [" << LaplaceToString(CT::LAPLACE_TYPE_[0]) << " "
             << LaplaceToString(CT::SMOOTH_VMULT_[0]) << " "
             << SmootherToString(CT::SMOOTH_INV_[0]) << " "
             << LocalSolverToString(CT::LOCAL_SOLVER_[0]) << "]\n";

    unsigned int index = k;

    info_table[index].add_value("level", triangulation.n_global_levels());
    info_table[index].add_value("cells", triangulation.n_global_active_cells());
    info_table[index].add_value("dofs", dof_handler.n_dofs());

    std::vector<PSMF::SolverData> comp_data = solver.static_comp();
    for (auto &data : comp_data)
      {
        *pcout << data.print_comp();

        auto times = data.solver_name + "[s]";
        auto perfs = data.solver_name + "Perf[Dof/s]";

        info_table[index].add_value(times, data.timing);
        info_table[index].add_value(perfs, data.perf);

        if (call_count == 0)
          {
            info_table[index].set_scientific(times, true);
            info_table[index].set_precision(times, 3);
            info_table[index].set_scientific(perfs, true);
            info_table[index].set_precision(perfs, 3);

            info_table[index].add_column_to_supercolumn(times,
                                                        data.solver_name);
            info_table[index].add_column_to_supercolumn(perfs,
                                                        data.solver_name);
          }
      }

    *pcout << std::endl;

    std::vector<PSMF::SolverData> solver_data = solver.solve();
    for (auto &data : solver_data)
      {
        *pcout << data.print_solver();

        auto it    = data.solver_name + "it";
        auto step  = data.solver_name + "step";
        auto times = data.solver_name + "[s]";
        auto mem   = data.solver_name + "Mem Usage[MB]";

        info_table[index].add_value(it, data.n_iteration);
        info_table[index].add_value(step, data.n_step);
        info_table[index].add_value(times, data.timing);
        info_table[index].add_value(mem, data.mem_usage);

        if (call_count == 0)
          {
            info_table[index].set_scientific(times, true);
            info_table[index].set_precision(times, 3);

            info_table[index].add_column_to_supercolumn(it, data.solver_name);
            info_table[index].add_column_to_supercolumn(step, data.solver_name);
            info_table[index].add_column_to_supercolumn(times,
                                                        data.solver_name);
            info_table[index].add_column_to_supercolumn(mem, data.solver_name);
          }
      }

    if (CT::SETS_ == "error_analysis")
      {
        auto solution = solver.get_solution();

        LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
                                               solution_host(solution.size());
        LinearAlgebra::ReadWriteVector<double> rw_vector(solution.size());
        rw_vector.import(solution, VectorOperation::insert);
        solution_host.import(rw_vector, VectorOperation::insert);
        ghost_solution_host = solution_host;
        constraints.distribute(ghost_solution_host);

        const auto [l2_error, H1_error] = compute_error();

        *pcout << "L2 error: " << l2_error << std::endl
               << "H1 error: " << H1_error << std::endl
               << std::endl;

        // ghost_solution_host.print(std::cout);

        info_table[index].add_value("L2_error", l2_error);
        info_table[index].set_scientific("L2_error", true);
        info_table[index].set_precision("L2_error", 3);

        info_table[index].evaluate_convergence_rates(
          "L2_error", "dofs", ConvergenceTable::reduction_rate_log2, dim);

        info_table[index].add_value("H1_error", H1_error);
        info_table[index].set_scientific("H1_error", true);
        info_table[index].set_precision("H1_error", 3);

        info_table[index].evaluate_convergence_rates(
          "H1_error", "dofs", ConvergenceTable::reduction_rate_log2, dim);
      }
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::solve_mg(unsigned int n_mg_cycles)
  {
    static unsigned int call_count = 0;

    using LA = PSMF::LaplaceVariant;
    using SM = PSMF::SmootherVariant;

    // do_solve<CT::LOCAL_SOLVER_[0],
    //          CT::LAPLACE_TYPE_[0],
    //          CT::SMOOTH_VMULT_[0],
    //          CT::SMOOTH_INV_[0]>(0, 0, 0, call_count);

    if (CT::LAPLACE_TYPE_.size() > 1)
      for (unsigned int k = 0; k < CT::LAPLACE_TYPE_.size(); ++k)
        {
          switch (CT::LAPLACE_TYPE_[k])
            {
              case PSMF::LaplaceVariant::Basic:
                {
                  do_solve<CT::LOCAL_SOLVER_[0],
                           PSMF::LaplaceVariant::Basic,
                           CT::SMOOTH_VMULT_[0],
                           CT::SMOOTH_INV_[0]>(k, call_count);
                  break;
                }
              case PSMF::LaplaceVariant::ConflictFree:
                {
                  do_solve<CT::LOCAL_SOLVER_[0],
                           PSMF::LaplaceVariant::ConflictFree,
                           CT::SMOOTH_VMULT_[0],
                           CT::SMOOTH_INV_[0]>(k, call_count);
                  break;
                }
              case PSMF::LaplaceVariant::TensorCore:
                {
                  do_solve<CT::LOCAL_SOLVER_[0],
                           PSMF::LaplaceVariant::TensorCore,
                           CT::SMOOTH_VMULT_[0],
                           CT::SMOOTH_INV_[0]>(k, call_count);
                  break;
                }
              case PSMF::LaplaceVariant::TensorCoreMMA:
                {
                  do_solve<CT::LOCAL_SOLVER_[0],
                           PSMF::LaplaceVariant::TensorCoreMMA,
                           CT::SMOOTH_VMULT_[0],
                           CT::SMOOTH_INV_[0]>(k, call_count);
                  break;
                }
              case PSMF::LaplaceVariant::TensorCoreMMACF:
                {
                  do_solve<CT::LOCAL_SOLVER_[0],
                           PSMF::LaplaceVariant::TensorCoreMMACF,
                           CT::SMOOTH_VMULT_[0],
                           CT::SMOOTH_INV_[0]>(k, call_count);
                  break;
                }
              default:
                AssertThrow(false, ExcMessage("Invalid Smoother Variant."));
            }
        }
    else if (CT::SMOOTH_INV_.size() > 1)
      for (unsigned int k = 0; k < CT::SMOOTH_INV_.size(); ++k)
        {
          switch (CT::SMOOTH_INV_[k])
            {
              case PSMF::SmootherVariant::GLOBAL:
                {
#if SCHWARZTYPE != 0
                  AssertThrow(false, ExcMessage("Invalid Smoother Variant."));
#endif
                  do_solve<CT::LOCAL_SOLVER_[0],
                           CT::LAPLACE_TYPE_[0],
                           CT::SMOOTH_VMULT_[0],
                           PSMF::SmootherVariant::GLOBAL>(k, call_count);
                  break;
                }
              case PSMF::SmootherVariant::AVS_colored:
                {
#if SCHWARZTYPE != 1
                  AssertThrow(false, ExcMessage("Invalid Smoother Variant."));
#endif
                  do_solve<CT::LOCAL_SOLVER_[0],
                           CT::LAPLACE_TYPE_[0],
                           CT::SMOOTH_VMULT_[0],
                           PSMF::SmootherVariant::AVS_colored>(k, call_count);
                  break;
                }
              case PSMF::SmootherVariant::AVS_atomic:
                {
#if SCHWARZTYPE != 2
                  AssertThrow(false, ExcMessage("Invalid Smoother Variant."));
#endif
                  do_solve<CT::LOCAL_SOLVER_[0],
                           CT::LAPLACE_TYPE_[0],
                           CT::SMOOTH_VMULT_[0],
                           PSMF::SmootherVariant::AVS_atomic>(k, call_count);
                  break;
                }
              default:
                AssertThrow(false, ExcMessage("Invalid Smoother Variant."));
            }
        }
    else if (CT::LOCAL_SOLVER_.size() > 1)
      for (unsigned int k = 0; k < CT::LOCAL_SOLVER_.size(); ++k)
        {
          switch (CT::LOCAL_SOLVER_[k])
            {
              case PSMF::LocalSolverVariant::Exact:
                {
                  do_solve<PSMF::LocalSolverVariant::Exact,
                           CT::LAPLACE_TYPE_[0],
                           CT::SMOOTH_VMULT_[0],
                           CT::SMOOTH_INV_[0]>(k, call_count);
                  break;
                }
              case PSMF::LocalSolverVariant::Bila:
                {
                  do_solve<PSMF::LocalSolverVariant::Bila,
                           CT::LAPLACE_TYPE_[0],
                           CT::SMOOTH_VMULT_[0],
                           CT::SMOOTH_INV_[0]>(k, call_count);
                  break;
                }
              case PSMF::LocalSolverVariant::KSVD:
                {
                  do_solve<PSMF::LocalSolverVariant::KSVD,
                           CT::LAPLACE_TYPE_[0],
                           CT::SMOOTH_VMULT_[0],
                           CT::SMOOTH_INV_[0]>(k, call_count);
                  break;
                }
              case PSMF::LocalSolverVariant::NN:
                {
                  do_solve<PSMF::LocalSolverVariant::NN,
                           CT::LAPLACE_TYPE_[0],
                           CT::SMOOTH_VMULT_[0],
                           CT::SMOOTH_INV_[0]>(k, call_count);
                  break;
                }
              default:
                AssertThrow(false, ExcMessage("Invalid Smoother Variant."));
            }
        }
    else
      do_solve<CT::LOCAL_SOLVER_[0],
               CT::LAPLACE_TYPE_[0],
               CT::SMOOTH_VMULT_[0],
               CT::SMOOTH_INV_[0]>(0, call_count);



    call_count++;
  }

  template <int dim, int fe_degree>
  std::pair<double, double>
  LaplaceProblem<dim, fe_degree>::compute_error()
  {
    Vector<double> cellwise_norm(triangulation.n_active_cells());
    VectorTools::integrate_difference(
      dof_handler,
      ghost_solution_host,
      Solution<dim,
               full_number>(), // Functions::ZeroFunction<dim, full_number>(),
      cellwise_norm,
      QGauss<dim>(fe->degree + 1),
      VectorTools::L2_norm);
    const double global_norm =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_norm,
                                        VectorTools::L2_norm);

    Vector<double> cellwise_h1norm(triangulation.n_active_cells());
    VectorTools::integrate_difference(
      dof_handler,
      ghost_solution_host,
      Solution<dim,
               full_number>(), // Functions::ZeroFunction<dim, full_number>(),
      cellwise_h1norm,
      QGauss<dim>(fe->degree + 1),
      VectorTools::H1_seminorm);
    const double global_h1norm =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_h1norm,
                                        VectorTools::H1_seminorm);

    return std::make_pair(global_norm, global_h1norm);
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::run(const unsigned int n_cycles)
  {
    *pcout << Util::generic_info_to_fstring() << std::endl;

    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        *pcout << "Cycle " << cycle << std::endl;

        long long unsigned int n_dofs =
          std::pow(std::pow(2, triangulation.n_global_levels()) * fe_degree + 1,
                   dim);

        if (n_dofs > CT::MAX_SIZES_)
          {
            *pcout << "Max size reached, terminating." << std::endl;
            *pcout << std::endl;

            if (CT::LAPLACE_TYPE_.size() > 1)
              for (unsigned int k = 0; k < CT::LAPLACE_TYPE_.size(); ++k)
                {
                  unsigned int index = k;

                  std::ostringstream oss;

                  oss << "\n[" << LaplaceToString(CT::LAPLACE_TYPE_[k]) << " "
                      << LaplaceToString(CT::SMOOTH_VMULT_[0]) << " "
                      << SmootherToString(CT::SMOOTH_INV_[0]) << " "
                      << LocalSolverToString(CT::LOCAL_SOLVER_[0]) << "]\n";
                  info_table[index].write_text(oss);

                  *pcout << oss.str() << std::endl;
                }
            else if (CT::SMOOTH_INV_.size() > 1)
              for (unsigned int k = 0; k < CT::SMOOTH_INV_.size(); ++k)
                {
                  unsigned int index = k;

                  std::ostringstream oss;

                  oss << "\n[" << LaplaceToString(CT::LAPLACE_TYPE_[0]) << " "
                      << LaplaceToString(CT::SMOOTH_VMULT_[0]) << " "
                      << SmootherToString(CT::SMOOTH_INV_[k]) << " "
                      << LocalSolverToString(CT::LOCAL_SOLVER_[0]) << "]\n";
                  info_table[index].write_text(oss);

                  *pcout << oss.str() << std::endl;
                }
            else if (CT::LOCAL_SOLVER_.size() > 1)
              for (unsigned int k = 0; k < CT::LOCAL_SOLVER_.size(); ++k)
                {
                  unsigned int index = k;

                  std::ostringstream oss;

                  oss << "\n[" << LaplaceToString(CT::LAPLACE_TYPE_[0]) << " "
                      << LaplaceToString(CT::SMOOTH_VMULT_[0]) << " "
                      << SmootherToString(CT::SMOOTH_INV_[0]) << " "
                      << LocalSolverToString(CT::LOCAL_SOLVER_[k]) << "]\n";
                  info_table[index].write_text(oss);

                  *pcout << oss.str() << std::endl;
                }
            else
              for (unsigned int k = 0; k < 1; ++k)
                {
                  unsigned int index = k;

                  std::ostringstream oss;

                  oss << "\n[" << LaplaceToString(CT::LAPLACE_TYPE_[0]) << " "
                      << LaplaceToString(CT::SMOOTH_VMULT_[0]) << " "
                      << SmootherToString(CT::SMOOTH_INV_[0]) << " "
                      << LocalSolverToString(CT::LOCAL_SOLVER_[k]) << "]\n";
                  info_table[index].write_text(oss);

                  *pcout << oss.str() << std::endl;
                }

            return;
          }

        if (cycle == 0)
          {
            GridGenerator::hyper_cube(triangulation, 0., 1.);
            triangulation.refine_global(2);
          }
        else
          triangulation.refine_global(1);

        setup_system();
        assemble_mg();

        solve_mg(1);
        *pcout << std::endl;
      }
  }
} // namespace Step64
int
main(int argc, char *argv[])
{
  try
    {
      using namespace Step64;

      {
        int device_id = findCudaDevice(argc, (const char **)argv);
        AssertCuda(cudaSetDevice(device_id));
      }

      {
        LaplaceProblem<CT::DIMENSION_, CT::FE_DEGREE_> Laplace_problem;
        Laplace_problem.run(20);
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
