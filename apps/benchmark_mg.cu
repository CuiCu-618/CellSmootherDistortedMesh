/**
 * @file benchmark_mg.cu
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief Benchmarks on different MG componts.
 * @version 1.0
 * @date 2023-01-02
 *
 * @copyright Copyright (c) 2023
 *
 */


#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/cuda.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/matrix_free/cuda_fe_evaluation.h>
#include <deal.II/matrix_free/cuda_matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

#include "app_utilities.h"
#include "ct_parameter.h"
#include "cuda_mg_transfer.cuh"
#include "dealii/cuda_mf.cuh"
#include "laplace_operator.cuh"
#include "patch_base.cuh"
#include "patch_smoother.cuh"

using namespace dealii;

template <int dim, int fe_degree>
class LaplaceProblem
{
public:
  using full_number   = double;
  using vcycle_number = float;
  using MatrixTypeDP =
    PSMF::LaplaceOperator<dim, fe_degree, full_number, CT::DOF_LAYOUT_>;
  using MatrixTypeSP =
    PSMF::LaplaceOperator<dim, fe_degree, vcycle_number, CT::DOF_LAYOUT_>;
  using VectorTypeDP =
    LinearAlgebra::distributed::Vector<full_number, MemorySpace::CUDA>;
  using VectorTypeSP =
    LinearAlgebra::distributed::Vector<vcycle_number, MemorySpace::CUDA>;

  LaplaceProblem();
  ~LaplaceProblem();
  void
  run();

private:
  void
  setup_system();
  void
  bench_Ax();
  void
  bench_transfer();
  void
  bench_smooth();

  template <PSMF::SmootherVariant kernel>
  void
  do_smooth();

  Triangulation<dim>                  triangulation;
  std::shared_ptr<FiniteElement<dim>> fe;
  DoFHandler<dim>                     dof_handler;
  MappingQ1<dim>                      mapping;

  MGConstrainedDoFs mg_constrained_dofs;

  MatrixTypeDP system_matrix;
  MatrixTypeSP mg_matrices;

  VectorTypeDP solution_dp;
  VectorTypeDP system_rhs_dp;

  VectorTypeSP solution_sp;
  VectorTypeSP system_rhs_sp;

  double base_time_dp;
  double base_time_sp;

  unsigned int N;
  unsigned int n_mv;
  unsigned int n_dofs;

  std::fstream                        fout;
  std::shared_ptr<ConditionalOStream> pcout;

  std::array<ConvergenceTable, 4> info_table;
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
  , base_time_dp(0.)
  , base_time_sp(0.)
  , pcout(std::make_shared<ConditionalOStream>(std::cout, false))
{
  const auto filename = Util::get_filename();
  fout.open("Benchmark_" + filename + ".log", std::ios_base::out);
  pcout = std::make_shared<ConditionalOStream>(fout, true);
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
  dof_handler.distribute_dofs(*fe);
  dof_handler.distribute_mg_dofs();

  n_dofs = dof_handler.n_dofs();
  N      = 1;
  n_mv   = 1; // dof_handler.n_dofs() < 10000000 ? 100 : 20;

  const unsigned int nlevels = triangulation.n_global_levels();
  for (unsigned int level = 0; level < nlevels; ++level)
    Util::Lexicographic(dof_handler, level);
  Util::Lexicographic(dof_handler);

  *pcout << "Setting up dofs...\n";
  *pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << " = ("
         << ((int)std::pow(dof_handler.n_dofs() * 1.0000001, 1. / dim) - 1) /
              fe->degree
         << " x " << fe->degree << " + 1)^" << dim << std::endl;

  *pcout << "Setting up Matrix-Free...\n";
  // Initialization of Dirichlet boundaries
  std::set<types::boundary_id> dirichlet_boundary;
  dirichlet_boundary.insert(0);
  mg_constrained_dofs.initialize(dof_handler);
  mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                     dirichlet_boundary);
  MappingQ1<dim> mapping;
  unsigned int   maxlevel = triangulation.n_global_levels() - 1;

  IndexSet relevant_dofs;
  DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                maxlevel,
                                                relevant_dofs);
  // DP
  {
    AffineConstraints<full_number> level_constraints;
    level_constraints.reinit(relevant_dofs);
    level_constraints.add_lines(
      mg_constrained_dofs.get_boundary_indices(maxlevel));
    level_constraints.close();

    using MatrixFreeType = PSMF::MatrixFree<dim, full_number>;

    typename MatrixFreeType::AdditionalData additional_data;
    additional_data.mapping_update_flags =
      (update_values | update_gradients | update_JxW_values);
    additional_data.mg_level = maxlevel;
    std::shared_ptr<MatrixFreeType> mg_mf_storage_level(new MatrixFreeType());
    mg_mf_storage_level->reinit(mapping,
                                dof_handler,
                                level_constraints,
                                QGauss<1>(fe_degree + 1),
                                additional_data);

    system_matrix.initialize(mg_mf_storage_level);
  }
  // SP
  {
    AffineConstraints<vcycle_number> level_constraints;
    level_constraints.reinit(relevant_dofs);
    level_constraints.add_lines(
      mg_constrained_dofs.get_boundary_indices(maxlevel));
    level_constraints.close();

    using MatrixFreeType = PSMF::MatrixFree<dim, vcycle_number>;

    typename MatrixFreeType::AdditionalData additional_data;
    additional_data.mapping_update_flags =
      (update_values | update_gradients | update_JxW_values);
    additional_data.mg_level = maxlevel;
    std::shared_ptr<MatrixFreeType> mg_mf_storage_level(new MatrixFreeType());
    mg_mf_storage_level->reinit(mapping,
                                dof_handler,
                                level_constraints,
                                QGauss<1>(fe_degree + 1),
                                additional_data);

    mg_matrices.initialize(mg_mf_storage_level);
  }

  system_matrix.initialize_dof_vector(system_rhs_dp);
  mg_matrices.initialize_dof_vector(system_rhs_sp);

  solution_dp.reinit(system_rhs_dp);
  solution_sp.reinit(system_rhs_sp);

  system_rhs_dp = 1.;
  system_rhs_sp = 1.;
}
template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::bench_Ax()
{
  *pcout << "Benchmarking Mat-vec in double precision...\n";

  Timer  time;
  double best_time  = 1e10;
  double best_time2 = 1e10;

  for (unsigned int i = 0; i < N; ++i)
    {
      time.restart();
      for (unsigned int i = 0; i < n_mv; ++i)
        {
          system_matrix.vmult(solution_dp, system_rhs_dp);
          cudaDeviceSynchronize();
        }
      best_time = std::min(time.wall_time() / n_mv, best_time);
    }

  info_table[0].add_value("Name", "Mat-vec DP");
  info_table[0].add_value("Time[s]", best_time);
  info_table[0].add_value("Perf[Dof/s]", n_dofs / best_time);
  info_table[0].add_value("Speedup", 0);

  *pcout << "Benchmarking Mat-vec in single precision...\n";

  for (unsigned int i = 0; i < N; ++i)
    {
      time.restart();
      for (unsigned int i = 0; i < n_mv; ++i)
        {
          mg_matrices.vmult(solution_sp, system_rhs_sp);
          cudaDeviceSynchronize();
        }
      best_time2 = std::min(time.wall_time() / n_mv, best_time2);
    }

  info_table[0].add_value("Name", "Mat-vec SP");
  info_table[0].add_value("Time[s]", best_time2);
  info_table[0].add_value("Perf[Dof/s]", n_dofs / best_time2);
  info_table[0].add_value("Speedup", best_time / best_time2);
}
template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::bench_transfer()
{
  *pcout << "Benchmarking Transfer in double precision...\n";

  unsigned int max_level = triangulation.n_levels() - 1;
  VectorTypeDP u_coarse(dof_handler.n_dofs(max_level - 1));
  VectorTypeSP u_coarse_(dof_handler.n_dofs(max_level - 1));
  u_coarse  = 1.;
  u_coarse_ = 1.;

  PSMF::MGTransferCUDA<dim, full_number, CT::DOF_LAYOUT_> mg_transfer(
    mg_constrained_dofs);
  mg_transfer.build(dof_handler);

  Timer  time;
  double best_time  = 1e10;
  double best_time2 = 1e10;

  for (unsigned int i = 0; i < N; ++i)
    {
      time.restart();
      for (unsigned int i = 0; i < n_mv; ++i)
        {
          mg_transfer.prolongate(max_level, system_rhs_dp, u_coarse);
          mg_transfer.restrict_and_add(max_level, u_coarse, system_rhs_dp);
          cudaDeviceSynchronize();
        }
      best_time = std::min(time.wall_time() / n_mv, best_time);
    }

  info_table[1].add_value("Name", "Transfer DP");
  info_table[1].add_value("Time[s]", best_time);
  info_table[1].add_value("Perf[Dof/s]", n_dofs / best_time);
  info_table[1].add_value("Speedup", 0);

  *pcout << "Benchmarking Transfer in single precision...\n";

  PSMF::MGTransferCUDA<dim, vcycle_number, CT::DOF_LAYOUT_> mg_transfer_(
    mg_constrained_dofs);
  mg_transfer_.build(dof_handler);

  for (unsigned int i = 0; i < N; ++i)
    {
      time.restart();
      for (unsigned int i = 0; i < n_mv; ++i)
        {
          mg_transfer_.prolongate(max_level, system_rhs_sp, u_coarse_);
          mg_transfer_.restrict_and_add(max_level, u_coarse_, system_rhs_sp);
          cudaDeviceSynchronize();
        }
      best_time2 = std::min(time.wall_time() / n_mv, best_time2);
    }

  info_table[1].add_value("Name", "Transfer SP");
  info_table[1].add_value("Time[s]", best_time2);
  info_table[1].add_value("Perf[Dof/s]", n_dofs / best_time2);
  info_table[1].add_value("Speedup", best_time / best_time2);
}

template <int dim, int fe_degree>
template <PSMF::SmootherVariant kernel>
void
LaplaceProblem<dim, fe_degree>::do_smooth()
{
  *pcout << "Benchmarking Smoother in double precision...\n";

  // DP
  using SmootherTypeDP =
    PSMF::PatchSmoother<MatrixTypeDP, dim, fe_degree, kernel, CT::DOF_LAYOUT_>;
  SmootherTypeDP                          smooth_dp;
  typename SmootherTypeDP::AdditionalData additional_data;
  additional_data.relaxation         = 1.;
  additional_data.granularity_scheme = CT::GRANULARITY_;
  smooth_dp.initialize(system_matrix, additional_data);

  Timer  time;
  double best_time  = 1e10;
  double best_time2 = 1e10;

  for (unsigned int i = 0; i < N; ++i)
    {
      time.restart();
      for (unsigned int i = 0; i < n_mv; ++i)
        {
          smooth_dp.step(solution_dp, system_rhs_dp);
          cudaDeviceSynchronize();
        }
      best_time = std::min(time.wall_time() / n_mv, best_time);
    }

  info_table[2].add_value("Name",
                          std::string(SmootherToString(kernel)) + "_DP");
  info_table[2].add_value("Time[s]", best_time);
  info_table[2].add_value("Perf[Dof/s]", n_dofs / best_time);
  if (kernel == PSMF::SmootherVariant::GLOBAL)
    {
      base_time_dp = best_time;
      info_table[2].add_value("Speedup", 0);
    }
  else
    {
      info_table[2].add_value("Speedup", base_time_dp / best_time);
    }

  *pcout << "Benchmarking Smoother in single precision...\n";

  // SP
  using SmootherTypeSP =
    PSMF::PatchSmoother<MatrixTypeSP, dim, fe_degree, kernel, CT::DOF_LAYOUT_>;
  SmootherTypeSP                          smooth_sp;
  typename SmootherTypeSP::AdditionalData additional_data_;
  additional_data_.relaxation         = 1.;
  additional_data_.granularity_scheme = CT::GRANULARITY_;
  smooth_sp.initialize(mg_matrices, additional_data_);

  for (unsigned int i = 0; i < N; ++i)
    {
      time.restart();
      for (unsigned int i = 0; i < n_mv; ++i)
        {
          smooth_sp.step(solution_sp, system_rhs_sp);
          cudaDeviceSynchronize();
        }
      best_time2 = std::min(time.wall_time() / n_mv, best_time2);
    }

  info_table[3].add_value("Name",
                          std::string(SmootherToString(kernel)) + "_SP");
  info_table[3].add_value("Time[s]", best_time2);
  info_table[3].add_value("Perf[Dof/s]", n_dofs / best_time2);
  if (kernel == PSMF::SmootherVariant::GLOBAL)
    {
      base_time_sp = best_time2;
      info_table[3].add_value("Speedup", 0);
    }
  else
    {
      info_table[3].add_value("Speedup", base_time_sp / best_time2);
    }
}
template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::bench_smooth()
{
  for (unsigned int k = 0; k < CT::KERNEL_TYPE_.size(); ++k)
    switch (CT::KERNEL_TYPE_[k])
      {
        case PSMF::SmootherVariant::GLOBAL:
          do_smooth<PSMF::SmootherVariant::GLOBAL>();
          break;
        case PSMF::SmootherVariant::SEPERATE:
          do_smooth<PSMF::SmootherVariant::SEPERATE>();
          break;
        case PSMF::SmootherVariant::FUSED_BASE:
          do_smooth<PSMF::SmootherVariant::FUSED_BASE>();
          break;
        case PSMF::SmootherVariant::FUSED_L:
          do_smooth<PSMF::SmootherVariant::FUSED_L>();
          break;
        case PSMF::SmootherVariant::FUSED_3D:
          do_smooth<PSMF::SmootherVariant::FUSED_3D>();
          break;
        case PSMF::SmootherVariant::FUSED_CF:
          do_smooth<PSMF::SmootherVariant::FUSED_CF>();
          break;
        default:
          AssertThrow(false, ExcMessage("Invalid Smoother Variant."));
      }
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::run()
{
  *pcout << Util::generic_info_to_fstring() << std::endl;

  GridGenerator::hyper_cube(triangulation, 0., 1.);

  double n_dofs_1d = 0;
  if (dim == 2)
    n_dofs_1d = std::sqrt(CT::MAX_SIZES_);
  else if (dim == 3)
    n_dofs_1d = std::cbrt(CT::MAX_SIZES_);

  auto n_refinement =
    static_cast<unsigned int>(std::log2((n_dofs_1d - 1) / fe_degree));
  triangulation.refine_global(n_refinement);

  setup_system();
  bench_Ax();
  bench_transfer();
  bench_smooth();

  *pcout << std::endl;

  for (unsigned int k = 0; k < 4; ++k)
    {
      std::ostringstream oss;

      info_table[k].set_scientific("Time[s]", true);
      info_table[k].set_precision("Time[s]", 3);
      info_table[k].set_scientific("Perf[Dof/s]", true);
      info_table[k].set_precision("Perf[Dof/s]", 3);

      info_table[k].write_text(oss);
      *pcout << oss.str() << std::endl;
      *pcout << std::endl;
    }
}

int
main(int argc, char *argv[])
{
  try
    {
      {
        int device_id = findCudaDevice(argc, (const char **)argv);
        AssertCuda(cudaSetDevice(device_id));
      }

      {
        LaplaceProblem<CT::DIMENSION_, CT::FE_DEGREE_> Laplace_problem;
        Laplace_problem.run();
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