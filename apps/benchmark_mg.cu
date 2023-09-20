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

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

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
  using MatrixFreeDP  = PSMF::LevelVertexPatch<dim, fe_degree, full_number>;
  using MatrixFreeSP  = PSMF::LevelVertexPatch<dim, fe_degree, vcycle_number>;

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

  template <PSMF::LaplaceVariant kernel>
  void
  do_Ax();
  template <PSMF::LaplaceVariant smooth_vmult, PSMF::SmootherVariant smooth_inv>
  void
  do_smooth();
  size_t
  disp_gpu_usage();

  MPI_Comm                                  mpi_communicator;
  parallel::distributed::Triangulation<dim> triangulation;
  std::shared_ptr<FiniteElement<dim>>       fe;
  DoFHandler<dim>                           dof_handler;
  MappingQ1<dim>                            mapping;

  MGConstrainedDoFs mg_constrained_dofs;

  std::shared_ptr<MatrixFreeDP> mfdata_dp;
  std::shared_ptr<MatrixFreeSP> mfdata_sp;

  VectorTypeDP solution_dp;
  VectorTypeDP system_rhs_dp;

  VectorTypeSP solution_sp;
  VectorTypeSP system_rhs_sp;

  double base_time_dp;
  double base_time_sp;

  double max_gpu_usage;

  unsigned int N;
  unsigned int n_mv;
  unsigned int n_dofs;
  unsigned int maxlevel;

  std::fstream                        fout;
  std::shared_ptr<ConditionalOStream> pcout;

  std::array<ConvergenceTable, 5> info_table;
};

template <int dim, int fe_degree>
LaplaceProblem<dim, fe_degree>::LaplaceProblem()
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(
      MPI_COMM_WORLD,
      Triangulation<dim>::limit_level_difference_at_vertices,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy)
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
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      fout.open("Benchmark_" + filename + ".log", std::ios_base::out);
      pcout = std::make_shared<ConditionalOStream>(
        fout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0);
    }
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
  dof_handler.distribute_dofs(*fe);
  dof_handler.distribute_mg_dofs();

  n_dofs = dof_handler.n_dofs();
  N      = 5;
  n_mv   = dof_handler.n_dofs() < 10000000 ? 100 : 20;

  const unsigned int nlevels = triangulation.n_global_levels();

  auto n_replicate =
    CT::IS_REPLICATE_ ? Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) : 1;

  *pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << " = "
         << n_replicate << " x (" << (1 << (nlevels - 1)) << " x ("
         << fe->degree << " + 1))^" << dim << std::endl;

  Utilities::System::MemoryStats stats;
  Utilities::System::get_memory_stats(stats);
  Utilities::MPI::MinMaxAvg memory =
    Utilities::MPI::min_max_avg(stats.VmRSS / 1024., MPI_COMM_WORLD);

  *pcout << "Memory stats [MB]: " << memory.min << " [p" << memory.min_index
         << "] " << memory.avg << " " << memory.max << " [p" << memory.max_index
         << "]" << std::endl;

  *pcout << "Setting up Matrix-Free...\n";
  // Initialization of Dirichlet boundaries
  std::set<types::boundary_id> dirichlet_boundary;
  dirichlet_boundary.insert(0);
  mg_constrained_dofs.initialize(dof_handler);
  mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                     dirichlet_boundary);
  MappingQ1<dim> mapping;
  maxlevel = triangulation.n_global_levels() - 1;

  IndexSet relevant_dofs;
  DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                maxlevel,
                                                relevant_dofs);
  // DP
  {
    typename MatrixFreeDP::AdditionalData additional_data;
    additional_data.relaxation         = 1.;
    additional_data.use_coloring       = false;
    additional_data.patch_per_block    = CT::PATCH_PER_BLOCK_;
    additional_data.granularity_scheme = CT::GRANULARITY_;

    mfdata_dp = std::make_shared<MatrixFreeDP>();
    mfdata_dp->reinit(dof_handler, maxlevel, additional_data);
  }
  // SP
  {
    typename MatrixFreeSP::AdditionalData additional_data;
    additional_data.relaxation         = 1.;
    additional_data.use_coloring       = false;
    additional_data.patch_per_block    = CT::PATCH_PER_BLOCK_;
    additional_data.granularity_scheme = CT::GRANULARITY_;

    mfdata_sp = std::make_shared<MatrixFreeSP>();
    mfdata_sp->reinit(dof_handler, maxlevel, additional_data);
  }

  *pcout << "Memory stats [MB]: " << memory.min << " [p" << memory.min_index
         << "] " << memory.avg << " " << memory.max << " [p" << memory.max_index
         << "]" << std::endl;
}
template <int dim, int fe_degree>
template <PSMF::LaplaceVariant kernel>
void
LaplaceProblem<dim, fe_degree>::do_Ax()
{
  {
    PSMF::LaplaceOperator<dim, fe_degree, full_number, kernel> matrix_dp;
    matrix_dp.initialize(mfdata_dp, dof_handler, maxlevel);
    matrix_dp.initialize_dof_vector(system_rhs_dp);
    solution_dp.reinit(system_rhs_dp);

    system_rhs_dp = 1.;
    solution_dp   = 0.;

    Timer  time;
    double best_time = 1e10;

    for (unsigned int i = 0; i < N; ++i)
      {
        time.restart();
        for (unsigned int i = 0; i < n_mv; ++i)
          {
            matrix_dp.vmult(solution_dp, system_rhs_dp);
            cudaDeviceSynchronize();
          }
        best_time = std::min(time.wall_time() / n_mv, best_time);
      }

    Utilities::MPI::MinMaxAvg stat =
      Utilities::MPI::min_max_avg(best_time, MPI_COMM_WORLD);
    *pcout << "Vmult time " << stat.min << " [p" << stat.min_index << "] "
           << stat.avg << " " << stat.max << " [p" << stat.max_index << "]"
           << std::endl;
    // solution_dp.print(std::cout);
    // *pcout << solution_dp.l2_norm() << std::endl;

    info_table[0].add_value("Name",
                            std::string(LaplaceToString(kernel)) + " DP");
    info_table[0].add_value("Time[s]", best_time);
    info_table[0].add_value("Perf[Dof/s]", n_dofs / best_time);
    info_table[0].add_value("Perf[s/Dof]", best_time / n_dofs);
    info_table[0].add_value("Mem Usage", disp_gpu_usage());
  }

  {
    PSMF::LaplaceOperator<dim, fe_degree, vcycle_number, kernel> matrix_sp;
    matrix_sp.initialize(mfdata_sp, dof_handler, maxlevel);
    matrix_sp.initialize_dof_vector(system_rhs_sp);
    solution_sp.reinit(system_rhs_sp);

    system_rhs_sp = 1.;
    solution_sp   = 0.;

    Timer  time;
    double best_time = 1e10;

    for (unsigned int i = 0; i < N; ++i)
      {
        time.restart();
        for (unsigned int i = 0; i < n_mv; ++i)
          {
            matrix_sp.vmult(solution_sp, system_rhs_sp);
            cudaDeviceSynchronize();
          }
        best_time = std::min(time.wall_time() / n_mv, best_time);
      }

    info_table[1].add_value("Name",
                            std::string(LaplaceToString(kernel)) + " SP");
    info_table[1].add_value("Time[s]", best_time);
    info_table[1].add_value("Perf[Dof/s]", n_dofs / best_time);
    info_table[1].add_value("Perf[s/Dof]", best_time / n_dofs);
    info_table[1].add_value("Mem Usage", disp_gpu_usage());
  }
}
template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::bench_Ax()
{
  *pcout << "Benchmarking Mat-vec...\n";

  for (unsigned int k = 0; k < CT::LAPLACE_TYPE_.size(); ++k)
    switch (CT::LAPLACE_TYPE_[k])
      {
        case PSMF::LaplaceVariant::Basic:
          do_Ax<PSMF::LaplaceVariant::Basic>();
          break;
        case PSMF::LaplaceVariant::BasicCell:
          do_Ax<PSMF::LaplaceVariant::BasicCell>();
          break;
        case PSMF::LaplaceVariant::TensorCore:
          do_Ax<PSMF::LaplaceVariant::TensorCore>();
          break;
        case PSMF::LaplaceVariant::TensorCoreMMA:
          do_Ax<PSMF::LaplaceVariant::TensorCoreMMA>();
          break;
        case PSMF::LaplaceVariant::ConflictFree:
          do_Ax<PSMF::LaplaceVariant::ConflictFree>();
          break;
        default:
          AssertThrow(false, ExcMessage("Invalid Laplace Variant."));
      }
}
template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::bench_transfer()
{
  unsigned int max_level  = triangulation.n_levels() - 1;
  auto locally_owned_dofs = dof_handler.locally_owned_mg_dofs(max_level - 1);
  auto locally_relevant_dofs =
    DoFTools::extract_locally_relevant_level_dofs(dof_handler, max_level - 1);

  {
    *pcout << "Benchmarking Transfer in double precision...\n";

    VectorTypeDP u_coarse(dof_handler.n_dofs(max_level - 1));

    u_coarse.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    u_coarse = 1.;

    PSMF::MGTransferCUDA<dim, full_number, CT::DOF_LAYOUT_> mg_transfer(
      mg_constrained_dofs);
    mg_transfer.build(dof_handler);

    Timer  time;
    double best_time = 1e10;

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

    // *pcout << u_coarse.l2_norm() << std::endl;

    info_table[2].add_value("Name", "Transfer DP");
    info_table[2].add_value("Time[s]", best_time);
    info_table[2].add_value("Perf[Dof/s]", n_dofs / best_time);
    info_table[2].add_value("Perf[s/Dof]", best_time / n_dofs);
    info_table[2].add_value("Mem Usage", disp_gpu_usage());
  }

  {
    *pcout << "Benchmarking Transfer in single precision...\n";

    VectorTypeSP u_coarse_(dof_handler.n_dofs(max_level - 1));

    u_coarse_.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    u_coarse_ = 1.;

    PSMF::MGTransferCUDA<dim, vcycle_number, CT::DOF_LAYOUT_> mg_transfer_(
      mg_constrained_dofs);
    mg_transfer_.build(dof_handler);

    Timer  time;
    double best_time2 = 1e10;

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

    info_table[2].add_value("Name", "Transfer SP");
    info_table[2].add_value("Time[s]", best_time2);
    info_table[2].add_value("Perf[Dof/s]", n_dofs / best_time2);
    info_table[2].add_value("Perf[s/Dof]", best_time2 / n_dofs);
    info_table[2].add_value("Mem Usage", disp_gpu_usage());
  }
}

template <int dim, int fe_degree>
template <PSMF::LaplaceVariant smooth_vmult, PSMF::SmootherVariant smooth_inv>
void
LaplaceProblem<dim, fe_degree>::do_smooth()
{
  {
    // DP
    using MatrixTypeDP =
      PSMF::LaplaceOperator<dim, fe_degree, full_number, smooth_vmult>;
    MatrixTypeDP matrix_dp;
    matrix_dp.initialize(mfdata_dp, dof_handler, maxlevel);

    using SmootherTypeDP = PSMF::
      PatchSmoother<MatrixTypeDP, dim, fe_degree, smooth_vmult, smooth_inv>;
    SmootherTypeDP                          smooth_dp;
    typename SmootherTypeDP::AdditionalData smoother_data_dp;
    smoother_data_dp.data = mfdata_dp;

    smooth_dp.initialize(matrix_dp, smoother_data_dp);

    Timer  time;
    double best_time = 1e10;

    system_rhs_dp = 1.;

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

    Utilities::MPI::MinMaxAvg stat =
      Utilities::MPI::min_max_avg(best_time, MPI_COMM_WORLD);
    *pcout << "Smoother time " << stat.min << " [p" << stat.min_index << "] "
           << stat.avg << " " << stat.max << " [p" << stat.max_index << "]"
           << std::endl;
    // *pcout << solution_dp.l2_norm() << std::endl;

    info_table[3].add_value("Name",
                            std::string(LaplaceToString(smooth_vmult)) + " " +
                              std::string(SmootherToString(smooth_inv)) +
                              " DP");
    info_table[3].add_value("Time[s]", best_time);
    info_table[3].add_value("Perf[Dof/s]", n_dofs / best_time);
    info_table[3].add_value("Perf[s/Dof]", best_time / n_dofs);
    info_table[3].add_value("Mem Usage", disp_gpu_usage());
  }

  {
    // SP
    using MatrixTypeSP =
      PSMF::LaplaceOperator<dim, fe_degree, vcycle_number, smooth_vmult>;
    MatrixTypeSP matrix_sp;
    matrix_sp.initialize(mfdata_sp, dof_handler, maxlevel);

    using SmootherTypeSP = PSMF::
      PatchSmoother<MatrixTypeSP, dim, fe_degree, smooth_vmult, smooth_inv>;
    SmootherTypeSP                          smooth_sp;
    typename SmootherTypeSP::AdditionalData smoother_data_sp;
    smoother_data_sp.data = mfdata_sp;

    smooth_sp.initialize(matrix_sp, smoother_data_sp);

    system_rhs_sp = 1.;

    Timer  time;
    double best_time = 1e10;

    for (unsigned int i = 0; i < N; ++i)
      {
        time.restart();
        for (unsigned int i = 0; i < n_mv; ++i)
          {
            smooth_sp.step(solution_sp, system_rhs_sp);
            cudaDeviceSynchronize();
          }
        best_time = std::min(time.wall_time() / n_mv, best_time);
      }

    info_table[4].add_value("Name",
                            std::string(LaplaceToString(smooth_vmult)) + " " +
                              std::string(SmootherToString(smooth_inv)) +
                              " SP");
    info_table[4].add_value("Time[s]", best_time);
    info_table[4].add_value("Perf[Dof/s]", n_dofs / best_time);
    info_table[4].add_value("Perf[s/Dof]", best_time / n_dofs);
    info_table[4].add_value("Mem Usage", disp_gpu_usage());
  }
}
template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::bench_smooth()
{
  *pcout << "Benchmarking Smoothing...\n";

  for (unsigned int k = 0; k < CT::SMOOTH_VMULT_.size(); ++k)
    switch (CT::SMOOTH_VMULT_[k])
      {
        case PSMF::LaplaceVariant::Basic:
          for (unsigned int j = 0; j < CT::SMOOTH_INV_.size(); ++j)
            switch (CT::SMOOTH_INV_[j])
              {
                case PSMF::SmootherVariant::GLOBAL:
                  do_smooth<PSMF::LaplaceVariant::Basic,
                            PSMF::SmootherVariant::GLOBAL>();
                  break;
                case PSMF::SmootherVariant::FUSED_L:
                  do_smooth<PSMF::LaplaceVariant::Basic,
                            PSMF::SmootherVariant::FUSED_L>();
                  break;
                case PSMF::SmootherVariant::ConflictFree:
                  do_smooth<PSMF::LaplaceVariant::Basic,
                            PSMF::SmootherVariant::ConflictFree>();
                  break;
              }
          break;
        case PSMF::LaplaceVariant::ConflictFree:
          for (unsigned int j = 0; j < CT::SMOOTH_INV_.size(); ++j)
            switch (CT::SMOOTH_INV_[j])
              {
                case PSMF::SmootherVariant::GLOBAL:
                  do_smooth<PSMF::LaplaceVariant::ConflictFree,
                            PSMF::SmootherVariant::GLOBAL>();
                  break;
                case PSMF::SmootherVariant::FUSED_L:
                  do_smooth<PSMF::LaplaceVariant::ConflictFree,
                            PSMF::SmootherVariant::FUSED_L>();
                  break;
                case PSMF::SmootherVariant::ConflictFree:
                  do_smooth<PSMF::LaplaceVariant::ConflictFree,
                            PSMF::SmootherVariant::ConflictFree>();
                  break;
              }
          break;
        case PSMF::LaplaceVariant::TensorCore:
          for (unsigned int j = 0; j < CT::SMOOTH_INV_.size(); ++j)
            switch (CT::SMOOTH_INV_[j])
              {
                case PSMF::SmootherVariant::GLOBAL:
                  do_smooth<PSMF::LaplaceVariant::TensorCore,
                            PSMF::SmootherVariant::GLOBAL>();
                  break;
                case PSMF::SmootherVariant::FUSED_L:
                  do_smooth<PSMF::LaplaceVariant::TensorCore,
                            PSMF::SmootherVariant::FUSED_L>();
                  break;
                case PSMF::SmootherVariant::ConflictFree:
                  do_smooth<PSMF::LaplaceVariant::TensorCore,
                            PSMF::SmootherVariant::ConflictFree>();
                  break;
              }
          break;
        default:
          AssertThrow(false, ExcMessage("Invalid Smoother Variant."));
      }
}
template <int dim, int fe_degree>
size_t
LaplaceProblem<dim, fe_degree>::disp_gpu_usage()
{
  size_t free_mem, total_mem;
  AssertCuda(cudaMemGetInfo(&free_mem, &total_mem));

  unsigned int scale    = 1024 * 1024;
  size_t       used_mem = (total_mem - free_mem) / scale;

  // if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  //   *pcout << "Used memory  [MB]: " << (total_mem - free_mem) / scale
  //          << std::endl;
  //  << "free memory  [MB]: " << free_mem / scale << std::endl
  //  << "total memory [MB]: " << total_mem / scale << std::endl;

  return used_mem;
}
template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::run()
{
  *pcout << Util::generic_info_to_fstring() << std::endl;

  if (CT::IS_REPLICATE_)
    {
      auto n_replicate = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
      Tensor<1, dim> shift_vector;
      shift_vector[0] = 1;

      parallel::distributed::Triangulation<dim> tria(
        MPI_COMM_WORLD,
        Triangulation<dim>::limit_level_difference_at_vertices,
        parallel::distributed::Triangulation<
          dim>::construct_multigrid_hierarchy);

      GridGenerator::hyper_cube(tria, 0, 1);
      if (dim == 2)
        GridGenerator::replicate_triangulation(tria,
                                               {n_replicate, 1},
                                               triangulation);
      else if (dim == 3)
        GridGenerator::replicate_triangulation(tria,
                                               {n_replicate, 1, 1},
                                               triangulation);
    }
  else
    GridGenerator::hyper_cube(triangulation, 0., 1.);

  double n_dofs_1d = 0;
  if (dim == 2)
    n_dofs_1d = std::sqrt(CT::MAX_SIZES_);
  else if (dim == 3)
    n_dofs_1d = std::cbrt(CT::MAX_SIZES_);

  auto n_refinement =
    static_cast<unsigned int>(std::log2(n_dofs_1d / (fe_degree + 1)));
  triangulation.refine_global(n_refinement);

  setup_system();
  bench_Ax();
  bench_transfer();
  bench_smooth();

  *pcout << std::endl;

  for (unsigned int k = 0; k < 5; ++k)
    {
      std::ostringstream oss;

      info_table[k].set_scientific("Time[s]", true);
      info_table[k].set_precision("Time[s]", 3);
      info_table[k].set_scientific("Perf[Dof/s]", true);
      info_table[k].set_precision("Perf[Dof/s]", 3);
      info_table[k].set_scientific("Perf[s/Dof]", true);
      info_table[k].set_precision("Perf[s/Dof]", 3);

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