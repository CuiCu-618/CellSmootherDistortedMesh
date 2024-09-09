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

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <helper_cuda.h>

#include <fstream>

#include "app_utilities.h"
#include "ct_parameter.h"
#include "cuda_matrix_free.cuh"
#include "solver.cuh"
#include "utilities.cuh"


// -\delta u = f, u = 0 on \parital \Omege, f = 1.
// double percision

namespace Step64
{
  using namespace dealii;

  template <int dim, typename Number = double>
  class Solution : public Function<dim, Number>
  {
  public:
    virtual Number
    value(const Point<dim> &p, const unsigned int = 0) const override final
    {
      Number val = 1.;
      for (unsigned int d = 0; d < dim; ++d)
        val *= std::sin(numbers::PI * p[d]);
      return val;
    }

    virtual Tensor<1, dim, Number>
    gradient(const Point<dim> &p, const unsigned int = 0) const override final
    {
      Tensor<1, dim, Number> grad;
      for (unsigned int d = 0; d < dim; ++d)
        {
          grad[d] = 1.;
          for (unsigned int e = 0; e < dim; ++e)
            if (d == e)
              grad[d] *= numbers::PI * std::cos(numbers::PI * p[e]);
            else
              grad[d] *= std::sin(numbers::PI * p[e]);
        }
      return grad;
    }
  };

  template <int dim, typename Number = double>
  class RightHandSide : public Function<dim, Number>
  {
  public:
    virtual Number
    value(const Point<dim> &p, const unsigned int = 0) const override final
    {
      const Number arg = numbers::PI;
      Number       val = 1.;
      for (unsigned int d = 0; d < dim; ++d)
        val *= std::sin(arg * p[d]);
      return dim * arg * arg * val;
    }
  };

  template <int dim, int fe_degree>
  class LaplaceProblem
  {
  public:
    using full_number   = double;
    using vcycle_number = CT::VCYCLE_NUMBER_;
    using MatrixFree    = PSMF::MatrixFree<dim, full_number>;
    using VertexPatchDP = PSMF::LevelVertexPatch<dim, fe_degree, full_number>;
    using VertexPatchSP = PSMF::LevelVertexPatch<dim, fe_degree, vcycle_number>;
    using CellPatchDP   = PSMF::LevelCellPatch<dim, fe_degree, full_number>;

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
    solve_mg();
    std::pair<double, double>
    compute_error();

    template <PSMF::LaplaceVariant  laplace,
              PSMF::LaplaceVariant  smooth_vmult,
              PSMF::SmootherVariant smooth_inv>
    void
    do_solve(unsigned int k,
             unsigned int j,
             unsigned int i,
             unsigned int call_count);

    MPI_Comm                                  mpi_communicator;
    parallel::distributed::Triangulation<dim> triangulation;
    std::shared_ptr<FiniteElement<dim>>       fe;
    DoFHandler<dim>                           dof_handler;
    MappingQ1<dim>                            mapping;
    double                                    setup_time;

    std::vector<ConvergenceTable> info_table;

    std::fstream                        fout;
    std::shared_ptr<ConditionalOStream> pcout;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    MGLevelObject<std::shared_ptr<MatrixFree>>    level_mfdata;
    MGLevelObject<std::shared_ptr<VertexPatchDP>> patch_data_dp;
    MGLevelObject<std::shared_ptr<VertexPatchSP>> patch_data_sp;
    MGLevelObject<std::shared_ptr<CellPatchDP>>   cell_data_dp;
    MGConstrainedDoFs                             mg_constrained_dofs;
    AffineConstraints<double>                     constraints;

    PSMF::MGTransferCUDA<dim, vcycle_number> transfer;

    LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
      ghost_solution_host;
  };

  template <int dim, int fe_degree>
  LaplaceProblem<dim, fe_degree>::LaplaceProblem()
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(MPI_COMM_WORLD,
                    Triangulation<dim>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<
                      dim>::construct_multigrid_hierarchy)
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
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        fout.open(filename + ".log", std::ios_base::out);
        pcout = std::make_shared<ConditionalOStream>(
          fout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0);
      }

    info_table.resize(CT::LAPLACE_TYPE_.size() * CT::SMOOTH_VMULT_.size() *
                      CT::SMOOTH_INV_.size());
  }

  template <int dim, int fe_degree>
  LaplaceProblem<dim, fe_degree>::~LaplaceProblem()
  {
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
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

    auto n_replicate = CT::N_REPLICATE_;

    *pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << " = "
           << n_replicate << " x (" << (1 << (nlevels - 1)) << " x ("
           << fe->degree << " + 1))^" << dim << std::endl;

    constraints.clear();
    constraints.close();

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

    ghost_solution_host.reinit(locally_owned_dofs,
                               locally_relevant_dofs,
                               mpi_communicator);

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

    unsigned int minlevel = 0;
    unsigned int maxlevel = triangulation.n_global_levels() - 1;

    patch_data_dp.resize(minlevel, maxlevel);
    cell_data_dp.resize(minlevel, maxlevel);
    level_mfdata.resize(minlevel, maxlevel);

    if (std::is_same_v<vcycle_number, float>)
      patch_data_sp.resize(minlevel, maxlevel);

    Timer time;
    for (unsigned int level = minlevel; level <= maxlevel; ++level)
      {
        // double-precision matrix-free data
        {
          typename MatrixFree::AdditionalData additional_data;
          additional_data.mapping_update_flags =
            update_quadrature_points | update_values | update_gradients |
            update_JxW_values;
          additional_data.mapping_update_flags_inner_faces =
            update_values | update_gradients | update_JxW_values |
            update_normal_vectors;
          additional_data.mg_level    = level;
          additional_data.matrix_type = PSMF::MatrixType::level_matrix;

          level_mfdata[level] = std::make_shared<MatrixFree>();
          level_mfdata[level]->reinit(mapping,
                                      dof_handler,
                                      constraints,
                                      QGauss<1>(fe_degree + 1),
                                      IteratorFilters::LocallyOwnedLevelCell(),
                                      additional_data);
        }

        {
          typename VertexPatchDP::AdditionalData additional_data;
          additional_data.relaxation         = 1.;
          additional_data.use_coloring       = false;
          additional_data.patch_per_block    = CT::PATCH_PER_BLOCK_;
          additional_data.granularity_scheme = CT::GRANULARITY_;

          patch_data_dp[level] = std::make_shared<VertexPatchDP>();
          patch_data_dp[level]->reinit(dof_handler, level, additional_data);
        }

        // single-precision matrix-free data
        if (std::is_same_v<vcycle_number, float>)
          {
            typename VertexPatchSP::AdditionalData additional_data;
            additional_data.relaxation         = 1.;
            additional_data.use_coloring       = false;
            additional_data.patch_per_block    = CT::PATCH_PER_BLOCK_;
            additional_data.granularity_scheme = CT::GRANULARITY_;

            patch_data_sp[level] = std::make_shared<VertexPatchSP>();
            patch_data_sp[level]->reinit(dof_handler, level, additional_data);
          }

        {
          typename CellPatchDP::AdditionalData additional_data;
          additional_data.relaxation         = 1.;
          additional_data.cell_per_block     = CT::PATCH_PER_BLOCK_;
          additional_data.granularity_scheme = CT::GRANULARITY_;

          cell_data_dp[level] = std::make_shared<CellPatchDP>();
          cell_data_dp[level]->reinit(dof_handler, level, additional_data);
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
  template <PSMF::LaplaceVariant  laplace,
            PSMF::LaplaceVariant  smooth_vmult,
            PSMF::SmootherVariant smooth_inv>
  void
  LaplaceProblem<dim, fe_degree>::do_solve(unsigned int k,
                                           unsigned int j,
                                           unsigned int i,
                                           unsigned int call_count)
  {
    PSMF::MultigridSolver<dim,
                          fe_degree,
                          CT::DOF_LAYOUT_,
                          full_number,
                          laplace,
                          smooth_vmult,
                          smooth_inv,
                          vcycle_number>
      solver(dof_handler,
             level_mfdata,
             patch_data_dp,
             patch_data_sp,
             cell_data_dp,
             transfer,
             Solution<dim>(),
             RightHandSide<dim>(),
             pcout,
             1);

    *pcout << "\nMG with [" << LaplaceToString(CT::LAPLACE_TYPE_[k]) << " "
           << LaplaceToString(CT::SMOOTH_VMULT_[j]) << " "
           << SmootherToString(CT::SMOOTH_INV_[i]) << "]\n";

    unsigned int index =
      (k * CT::SMOOTH_VMULT_.size() + j) * CT::SMOOTH_INV_.size() + i;

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
        auto perf  = data.solver_name + "[s/Dof]";
        auto mem   = data.solver_name + "Mem Usage[MB]";

        info_table[index].add_value(it, data.n_iteration);
        info_table[index].add_value(step, data.n_step);
        info_table[index].add_value(times, data.timing);
        info_table[index].add_value(perf, data.timing / dof_handler.n_dofs());
        info_table[index].add_value(mem, data.mem_usage);

        if (call_count == 0)
          {
            info_table[index].set_scientific(times, true);
            info_table[index].set_precision(times, 3);

            info_table[index].set_scientific(perf, true);
            info_table[index].set_precision(perf, 3);

            info_table[index].add_column_to_supercolumn(it, data.solver_name);
            info_table[index].add_column_to_supercolumn(step, data.solver_name);
            info_table[index].add_column_to_supercolumn(times,
                                                        data.solver_name);
            info_table[index].add_column_to_supercolumn(perf, data.solver_name);
            info_table[index].add_column_to_supercolumn(mem, data.solver_name);
          }
      }

    if (CT::SETS_ == "error_analysis")
      {
        auto solution = solver.get_solution();

        LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);
        rw_vector.import(solution, VectorOperation::insert);

        ghost_solution_host.import(rw_vector, VectorOperation::insert);
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
  LaplaceProblem<dim, fe_degree>::solve_mg()
  {
    static unsigned int call_count = 0;

    using LA = PSMF::LaplaceVariant;
    using SM = PSMF::SmootherVariant;

    // do_solve<CT::LAPLACE_TYPE_[0], CT::SMOOTH_VMULT_[0], CT::SMOOTH_INV_[0]>(
    //   0, 0, 0, call_count);

    for (unsigned int k = 0; k < CT::SMOOTH_INV_.size(); ++k)
      {
        switch (CT::SMOOTH_INV_[k])
          {
            case PSMF::SmootherVariant::GLOBAL:
              {
                do_solve<CT::LAPLACE_TYPE_[0],
                         CT::SMOOTH_VMULT_[0],
                         PSMF::SmootherVariant::GLOBAL>(0, 0, k, call_count);
                break;
              }
            case PSMF::SmootherVariant::FUSED_L:
              {
                do_solve<CT::LAPLACE_TYPE_[0],
                         CT::SMOOTH_VMULT_[0],
                         PSMF::SmootherVariant::FUSED_L>(0, 0, k, call_count);
                break;
              }
            case PSMF::SmootherVariant::ConflictFree:
              {
                do_solve<CT::LAPLACE_TYPE_[0],
                         CT::SMOOTH_VMULT_[0],
                         PSMF::SmootherVariant::ConflictFree>(0,
                                                              0,
                                                              k,
                                                              call_count);
                break;
              }
            case PSMF::SmootherVariant::ExactRes:
              {
                do_solve<CT::LAPLACE_TYPE_[0],
                         CT::SMOOTH_VMULT_[0],
                         PSMF::SmootherVariant::ExactRes>(0, 0, k, call_count);
                break;
              }
            case PSMF::SmootherVariant::Chebyshev:
              {
                do_solve<CT::LAPLACE_TYPE_[0],
                         CT::SMOOTH_VMULT_[0],
                         PSMF::SmootherVariant::Chebyshev>(0, 0, k, call_count);
                break;
              }
            case PSMF::SmootherVariant::MCS:
              {
                do_solve<CT::LAPLACE_TYPE_[0],
                         CT::SMOOTH_VMULT_[0],
                         PSMF::SmootherVariant::MCS>(0, 0, k, call_count);
                break;
              }
            default:
              AssertThrow(false, ExcMessage("Invalid Smoother Variant."));
          }
      }
    // for (unsigned int k = 0; k < CT::LAPLACE_TYPE_.size(); ++k)
    //   for (unsigned int j = 0; j < CT::SMOOTH_VMULT_.size(); ++j)
    //     for (unsigned int i = 0; i < CT::SMOOTH_INV_.size(); ++i)
    //       {
    //         if (LAPLACE_TYPE_[i] == LA::Basic)

    //       }



    call_count++;
  }


  template <int dim, int fe_degree>
  std::pair<double, double>
  LaplaceProblem<dim, fe_degree>::compute_error()
  {
    Vector<double> cellwise_norm(triangulation.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      ghost_solution_host,
                                      Solution<dim>(),
                                      cellwise_norm,
                                      QGauss<dim>(fe->degree + 1),
                                      VectorTools::L2_norm);
    const double global_norm = std::sqrt(
      Utilities::MPI::sum(cellwise_norm.norm_sqr(), mpi_communicator));

    Vector<double> cellwise_h1norm(triangulation.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      ghost_solution_host,
                                      Solution<dim>(),
                                      cellwise_h1norm,
                                      QGauss<dim>(fe->degree + 1),
                                      VectorTools::H1_seminorm);
    const double global_h1norm = std::sqrt(
      Utilities::MPI::sum(cellwise_h1norm.norm_sqr(), mpi_communicator));

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

        long long unsigned int n_dofs = std::pow(
          std::pow(2, triangulation.n_global_levels()) * (fe_degree + 1), dim);

        if (n_dofs > CT::MAX_SIZES_)
          {
            *pcout << "Max size reached, terminating." << std::endl;
            *pcout << std::endl;

            break;
          }

        if (cycle == 0)
          {
            // auto n_replicate =
            //   Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

            parallel::distributed::Triangulation<dim> tria(
              MPI_COMM_WORLD,
              Triangulation<dim>::limit_level_difference_at_vertices,
              parallel::distributed::Triangulation<
                dim>::construct_multigrid_hierarchy);

            GridGenerator::hyper_cube(tria, 0, 1);
            if (dim == 2)
              GridGenerator::replicate_triangulation(tria,
                                                     {CT::N_REPLICATE_, 1},
                                                     triangulation);
            else if (dim == 3)
              GridGenerator::replicate_triangulation(tria,
                                                     {CT::N_REPLICATE_, 1, 1},
                                                     triangulation);

            triangulation.refine_global(2);
          }
        else
          triangulation.refine_global(1);

        setup_system();
        assemble_mg();

        solve_mg();
        *pcout << std::endl;
      }

    {
      for (unsigned int k = 0; k < CT::LAPLACE_TYPE_.size(); ++k)
        for (unsigned int j = 0; j < CT::SMOOTH_VMULT_.size(); ++j)
          for (unsigned int i = 0; i < CT::SMOOTH_INV_.size(); ++i)
            {
              unsigned int index =
                (k * CT::SMOOTH_VMULT_.size() + j) * CT::SMOOTH_INV_.size() + i;

              std::ostringstream oss;

              oss << "\n[" << LaplaceToString(CT::LAPLACE_TYPE_[k]) << " "
                  << LaplaceToString(CT::SMOOTH_VMULT_[j]) << " "
                  << SmootherToString(CT::SMOOTH_INV_[i]) << "]\n";
              info_table[index].write_text(oss);

              *pcout << oss.str() << std::endl;
            }
    }
  }
} // namespace Step64
int
main(int argc, char *argv[])
{
  try
    {
      using namespace Step64;

      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      {
        int         n_devices       = 0;
        cudaError_t cuda_error_code = cudaGetDeviceCount(&n_devices);
        AssertCuda(cuda_error_code);
        const unsigned int my_mpi_id =
          Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        const int device_id = my_mpi_id % n_devices;
        cuda_error_code     = cudaSetDevice(device_id);
        AssertCuda(cuda_error_code);
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
