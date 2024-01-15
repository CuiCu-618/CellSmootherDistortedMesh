/**
 * Created by Cu Cui on 2022/4/15.
 */

// test coloring patches with mpi
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

template <int dim, int fe_degree>
class test
{
public:
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::level_cell_iterator>;
  using PatchIterator =
    typename std::vector<std::vector<CellFilter>>::const_iterator;

  test(unsigned int n_refinements);
  void
  run();
  void
  setup_system();
  void
  coloring();
  void
  output_results(unsigned int cycle) const;

private:
  unsigned int                              n_refinements;
  MPI_Comm                                  mpi_communicator;
  parallel::distributed::Triangulation<dim> triangulation;
  FE_Q<dim>                                 fe;
  DoFHandler<dim>                           dof_handler;
  IndexSet                                  locally_owned_dofs;
  IndexSet                                  locally_relevant_dofs;
  AffineConstraints<double>                 constraints;
  ConditionalOStream                        pcout;
  TimerOutput                               computing_timer;
};

template <int dim, int fe_degree>
test<dim, fe_degree>::test(const unsigned int n_refinements)
  : n_refinements(n_refinements)
  , mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(
                    Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening))
  , fe(fe_degree)
  , dof_handler(triangulation)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::never,
                    TimerOutput::wall_times)
{}
template <int dim, int fe_degree>
void
test<dim, fe_degree>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints);
  constraints.close();
}
template <int dim, int fe_degree>
void
test<dim, fe_degree>::output_results(const unsigned int cycle) const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  LinearAlgebra::distributed::Vector<float> marker;
  marker.reinit(dof_handler.locally_owned_dofs(),
                locally_relevant_dofs,
                MPI_COMM_WORLD);
  marker = 0;


  data_out.build_patches();
  data_out.write_vtu_with_pvtu_record(
    "./", "solution", cycle, mpi_communicator, 2, 8);
}
template <int dim, int fe_degree>
void
test<dim, fe_degree>::run()
{
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(6);

  setup_system();

  pcout << "   Number of active cells:       "
        << triangulation.n_global_active_cells() << std::endl
        << "   Number of degrees of freedom: " << dof_handler.n_dofs()
        << std::endl;
  //   coloring();
  output_results(0);
}
template <int dim, int fe_degree>
void
test<dim, fe_degree>::coloring()
{
  LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
    ghost_solution_host;
  ghost_solution_host.reinit(locally_owned_dofs,
                             locally_relevant_dofs,
                             mpi_communicator);
  //  locally_owned_dofs.print(std::cout);
  //  locally_relevant_dofs.print(std::cout);
  const unsigned int                   dofs_per_cell = fe.n_dofs_per_cell();
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  double                              *data = ghost_solution_host.get_values();
  auto part = ghost_solution_host.get_partitioner();
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          data[part->global_to_local(local_dof_indices[i])] = 1;
      }
  ghost_solution_host.compress(VectorOperation::add);
  ghost_solution_host.print(std::cout);

  Vector<double> vec(dof_handler.n_dofs());

  unsigned int offset =
    dof_handler.n_dofs() / Utilities::MPI::n_mpi_processes(mpi_communicator);
  for (unsigned int i = 0; i < offset + 1; ++i)
    vec(i + Utilities::MPI::this_mpi_process(mpi_communicator) * offset) = 1;

  vec.print(std::cout);
  std::cout << Utilities::MPI::this_mpi_process(mpi_communicator) << std::endl;

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      auto *buff = new double[dof_handler.n_dofs()];
      for (unsigned int r = 1;
           r < Utilities::MPI::n_mpi_processes(mpi_communicator);
           ++r)
        MPI_Recv(buff,
                 dof_handler.n_dofs(),
                 MPI_DOUBLE,
                 r,
                 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

      for (unsigned int i = 0; i < vec.size(); ++i)
        vec[i] += buff[i];
      vec.print(std::cout);
      delete[] buff;
    }
  else
    {
      MPI_Send(
        vec.begin(), dof_handler.n_dofs(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

  IteratorFilters::LocallyOwnedCell locally_owned_cell_filter;
  CellFilter begin(locally_owned_cell_filter, dof_handler.begin_active());
  CellFilter end(locally_owned_cell_filter, dof_handler.end());
}


int
main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      MPI_Comm           mpi_communicator(MPI_COMM_WORLD);
      ConditionalOStream pcout(
        std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0));

      pcout << "Running tests on "
            << Utilities::MPI::n_mpi_processes(mpi_communicator)
            << " MPI rank(s)..." << std::endl;

      test<2, 1> t1(2);
      t1.run();
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