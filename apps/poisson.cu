/**
 * Created by Cu Cui on 2022/12/25.
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/cuda.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_tools.h>

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
#include "solver.cuh"
#include "utilities.cuh"


// -\delta u = f, u = 0 on \parital \Omege, f = 1.
// double percision

namespace Step64
{
  using namespace dealii;

  template <int dim, typename Number>
  class Solution : public Function<dim, Number>
  {
  public:
    virtual Number
    value(const Point<dim> &, const unsigned int = 0) const override final
    {
      return 0.;
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
              grad[d] *= -numbers::PI * std::cos(numbers::PI * p[e]);
            else
              grad[d] *= std::sin(numbers::PI * p[e]);
        }
      return grad;
    }
  };

  template <int dim, typename Number>
  class RightHandSide : public Function<dim, Number>
  {
  public:
    virtual Number
    value(const Point<dim> &, const unsigned int = 0) const override final
    {
      return 1.;
    }
  };

  template <int dim, int fe_degree>
  class LaplaceProblem
  {
  public:
    using full_number   = double;
    using vcycle_number = CT::VCYCLE_NUMBER_;
    using MatrixTypeDP =
      PSMF::LaplaceOperator<dim, fe_degree, full_number, CT::DOF_LAYOUT_>;
    using MatrixTypeSP =
      PSMF::LaplaceOperator<dim, fe_degree, vcycle_number, CT::DOF_LAYOUT_>;

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

    Triangulation<dim>                  triangulation;
    std::shared_ptr<FiniteElement<dim>> fe;
    DoFHandler<dim>                     dof_handler;
    MappingQ1<dim>                      mapping;
    double                              setup_time;

    std::vector<ConvergenceTable> info_table;

    std::fstream                        fout;
    std::shared_ptr<ConditionalOStream> pcout;

    MGLevelObject<MatrixTypeDP> matrix_dp;
    MGLevelObject<MatrixTypeSP> matrix;
    MGConstrainedDoFs           mg_constrained_dofs;
  };

  template <int dim, int fe_degree>
  LaplaceProblem<dim, fe_degree>::LaplaceProblem()
    : triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
    , fe([&]() -> std::shared_ptr<FiniteElement<dim>> {
      if (CT::DOF_LAYOUT_ == PSMF::DoFLayout::Q)
        return std::make_shared<FE_Q<dim>>(fe_degree);
      else if (CT::DOF_LAYOUT_ == PSMF::DoFLayout::DGQ)
        return std::make_shared<FE_DGQ<dim>>(fe_degree);
      return std::shared_ptr<FiniteElement<dim>>();
    }())
    , dof_handler(triangulation)
    , setup_time(0.)
    , pcout(std::make_shared<ConditionalOStream>(std::cout, false))
  {
    const auto filename = Util::get_filename();
    fout.open(filename + ".log", std::ios_base::out);
    pcout = std::make_shared<ConditionalOStream>(fout, true);

    info_table.resize(CT::KERNEL_TYPE_.size());
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

    matrix_dp.resize(1, maxlevel);

    if (std::is_same_v<vcycle_number, float>)
      matrix.resize(1, maxlevel);

    Timer time;
    for (unsigned int level = minlevel; level <= maxlevel; ++level)
      {
        IndexSet relevant_dofs;
        DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                      level,
                                                      relevant_dofs);
        // double-precision matrix-free data
        {
          AffineConstraints<full_number> level_constraints;
          level_constraints.reinit(relevant_dofs);
          level_constraints.add_lines(
            mg_constrained_dofs.get_boundary_indices(level));
          level_constraints.close();

          using MatrixFreeType = PSMF::MatrixFree<dim, full_number>;

          typename MatrixFreeType::AdditionalData additional_data;
          additional_data.mapping_update_flags =
            (update_values | update_gradients | update_JxW_values);
          additional_data.mg_level = level;
          std::shared_ptr<MatrixFreeType> mg_mf_storage_level(
            new MatrixFreeType());
          mg_mf_storage_level->reinit(mapping,
                                      dof_handler,
                                      level_constraints,
                                      QGauss<1>(fe_degree + 1),
                                      additional_data);

          matrix_dp[level].initialize(mg_mf_storage_level);
        }

        // single-precision matrix-free data
        if (std::is_same_v<vcycle_number, float>)
          {
            AffineConstraints<vcycle_number> level_constraints;
            level_constraints.reinit(relevant_dofs);
            level_constraints.add_lines(
              mg_constrained_dofs.get_boundary_indices(level));
            level_constraints.close();

            using MatrixFreeType = PSMF::MatrixFree<dim, vcycle_number>;

            typename MatrixFreeType::AdditionalData additional_data;
            additional_data.mapping_update_flags =
              update_gradients | update_JxW_values;
            additional_data.mg_level = level;
            std::shared_ptr<MatrixFreeType> mg_mf_storage_level(
              new MatrixFreeType());
            mg_mf_storage_level->reinit(mapping,
                                        dof_handler,
                                        level_constraints,
                                        QGauss<1>(fe_degree + 1),
                                        additional_data);

            matrix[level].initialize(mg_mf_storage_level);
          }
      }

    *pcout << "Matrix-free setup time: " << time.wall_time() << "s"
           << std::endl;
  }
  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::solve_mg(unsigned int n_mg_cycles)
  {
    Timer time;

    PSMF::MGTransferCUDA<dim, full_number, CT::DOF_LAYOUT_>   transfer_dp;
    PSMF::MGTransferCUDA<dim, vcycle_number, CT::DOF_LAYOUT_> transfer;

    if (std::is_same_v<vcycle_number, float>)
      {
        transfer_dp.build(dof_handler);
        transfer.initialize_constraints(mg_constrained_dofs);
        transfer.build(dof_handler);
      }
    else if (std::is_same_v<vcycle_number, double>)
      {
        transfer_dp.initialize_constraints(mg_constrained_dofs);
        transfer_dp.build(dof_handler);
      }
    else
      AssertThrow(false, ExcMessage("Invalid V-cycle number type."));

    *pcout << "MG transfer setup time: " << time.wall_time() << "s"
           << std::endl;

    static unsigned int call_count = 0;

    auto do_solver = [&](auto solver, unsigned int k) {
      *pcout << "\nMG with smooth variant ["
             << SmootherToString(CT::KERNEL_TYPE_[k]) << "]\n";

      info_table[k].add_value("level", triangulation.n_global_levels());
      info_table[k].add_value("cells", triangulation.n_global_active_cells());
      info_table[k].add_value("dofs", dof_handler.n_dofs());

      std::vector<PSMF::SolverData> comp_data = solver.static_comp();
      for (auto &data : comp_data)
        {
          *pcout << data.print_comp();

          auto times = data.solver_name + "[s]";
          auto perfs = data.solver_name + "Perf[Dof/s]";

          info_table[k].add_value(times, data.timing);
          info_table[k].add_value(perfs, data.perf);

          if (call_count == 0)
            {
              info_table[k].set_scientific(times, true);
              info_table[k].set_precision(times, 3);
              info_table[k].set_scientific(perfs, true);
              info_table[k].set_precision(perfs, 3);

              info_table[k].add_column_to_supercolumn(times, data.solver_name);
              info_table[k].add_column_to_supercolumn(perfs, data.solver_name);
            }
        }

      *pcout << std::endl;

      std::vector<PSMF::SolverData> solver_data = solver.solve();
      for (auto &data : solver_data)
        {
          *pcout << data.print_solver();

          auto it    = data.solver_name + "it";
          auto times = data.solver_name + "[s]";
          auto mem   = data.solver_name + "Mem Usage[MB]";

          info_table[k].add_value(it, data.n_iteration);
          info_table[k].add_value(times, data.timing);
          info_table[k].add_value(mem, data.mem_usage);

          if (call_count == 0)
            {
              info_table[k].set_scientific(times, true);
              info_table[k].set_precision(times, 3);

              info_table[k].add_column_to_supercolumn(it, data.solver_name);
              info_table[k].add_column_to_supercolumn(times, data.solver_name);
              info_table[k].add_column_to_supercolumn(mem, data.solver_name);
            }
        }
    };

    for (unsigned int k = 0; k < CT::KERNEL_TYPE_.size(); ++k)
      {
        switch (CT::KERNEL_TYPE_[k])
          {
            case PSMF::SmootherVariant::GLOBAL:
              {
                PSMF::MultigridSolver<dim,
                                      fe_degree,
                                      CT::DOF_LAYOUT_,
                                      full_number,
                                      PSMF::SmootherVariant::GLOBAL,
                                      vcycle_number>
                  solver(dof_handler,
                         matrix_dp,
                         matrix,
                         transfer_dp,
                         transfer,
                         Functions::ZeroFunction<dim, full_number>(),
                         Functions::ConstantFunction<dim, full_number>(1.),
                         pcout,
                         n_mg_cycles);
                do_solver(solver, k);
                break;
              }
            case PSMF::SmootherVariant::SEPERATE:
              {
                PSMF::MultigridSolver<dim,
                                      fe_degree,
                                      CT::DOF_LAYOUT_,
                                      full_number,
                                      PSMF::SmootherVariant::SEPERATE,
                                      vcycle_number>
                  solver(dof_handler,
                         matrix_dp,
                         matrix,
                         transfer_dp,
                         transfer,
                         Functions::ZeroFunction<dim, full_number>(),
                         Functions::ConstantFunction<dim, full_number>(1.),
                         pcout,
                         n_mg_cycles);
                do_solver(solver, k);
                break;
              }
            case PSMF::SmootherVariant::FUSED_BASE:
              {
                PSMF::MultigridSolver<dim,
                                      fe_degree,
                                      CT::DOF_LAYOUT_,
                                      full_number,
                                      PSMF::SmootherVariant::FUSED_BASE,
                                      vcycle_number>
                  solver(dof_handler,
                         matrix_dp,
                         matrix,
                         transfer_dp,
                         transfer,
                         Functions::ZeroFunction<dim, full_number>(),
                         Functions::ConstantFunction<dim, full_number>(1.),
                         pcout,
                         n_mg_cycles);
                do_solver(solver, k);
                break;
              }
            case PSMF::SmootherVariant::FUSED_L:
              {
                PSMF::MultigridSolver<dim,
                                      fe_degree,
                                      CT::DOF_LAYOUT_,
                                      full_number,
                                      PSMF::SmootherVariant::FUSED_L,
                                      vcycle_number>
                  solver(dof_handler,
                         matrix_dp,
                         matrix,
                         transfer_dp,
                         transfer,
                         Functions::ZeroFunction<dim, full_number>(),
                         Functions::ConstantFunction<dim, full_number>(1.),
                         pcout,
                         n_mg_cycles);
                do_solver(solver, k);
                break;
              }
            case PSMF::SmootherVariant::FUSED_3D:
              {
                PSMF::MultigridSolver<dim,
                                      fe_degree,
                                      CT::DOF_LAYOUT_,
                                      full_number,
                                      PSMF::SmootherVariant::FUSED_3D,
                                      vcycle_number>
                  solver(dof_handler,
                         matrix_dp,
                         matrix,
                         transfer_dp,
                         transfer,
                         Functions::ZeroFunction<dim, full_number>(),
                         Functions::ConstantFunction<dim, full_number>(1.),
                         pcout,
                         n_mg_cycles);
                do_solver(solver, k);
                break;
              }
            case PSMF::SmootherVariant::FUSED_CF:
              {
                PSMF::MultigridSolver<dim,
                                      fe_degree,
                                      CT::DOF_LAYOUT_,
                                      full_number,
                                      PSMF::SmootherVariant::FUSED_CF,
                                      vcycle_number>
                  solver(dof_handler,
                         matrix_dp,
                         matrix,
                         transfer_dp,
                         transfer,
                         Functions::ZeroFunction<dim, full_number>(),
                         Functions::ConstantFunction<dim, full_number>(1.),
                         pcout,
                         n_mg_cycles);
                do_solver(solver, k);
                break;
              }
            default:
              AssertThrow(false, ExcMessage("Invalid Smoother Variant."));
          }
      }

    call_count++;
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::run(const unsigned int n_cycles)
  {
    *pcout << Util::generic_info_to_fstring() << std::endl;

    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        smoother_mem = 0;

        *pcout << "Cycle " << cycle << std::endl;

        long long unsigned int n_dofs =
          std::pow(std::pow(2, triangulation.n_global_levels()) * fe_degree + 1,
                   dim);

        if (n_dofs > CT::MAX_SIZES_)
          {
            *pcout << "Max size reached, terminating." << std::endl;
            *pcout << std::endl;

            for (unsigned int k = 0; k < CT::KERNEL_TYPE_.size(); ++k)
              {
                std::ostringstream oss;

                oss << "\n[" << SmootherToString(CT::KERNEL_TYPE_[k]) << "]\n";
                info_table[k].write_text(oss);

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