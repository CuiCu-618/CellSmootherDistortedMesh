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

__global__ void
write_ghost(double *vec, int idx)
{
  vec[idx] += 6000000;
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

      MPI_Comm           mpi_communicator(MPI_COMM_WORLD);
      ConditionalOStream pcout(
        std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0));

      pcout << "Running tests on "
            << Utilities::MPI::n_mpi_processes(mpi_communicator)
            << " MPI rank(s)..." << std::endl;

      LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> gpu_vec;

      IndexSet s(20);
      IndexSet g(20);
      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
          s.add_range(0, 10);
          g.add_index(11);
        }
      else
        {
          s.add_range(10, 20);
          g.add_index(9);
        }

      gpu_vec.reinit(s, g, mpi_communicator);

      gpu_vec = 1;
      std::cout << "Init: \n";
      gpu_vec.print(std::cout);

      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        write_ghost<<<1, 1>>>(gpu_vec.get_values(), 10);

      cudaDeviceSynchronize();

      std::cout << "before: \n";
      gpu_vec.print(std::cout);

      gpu_vec.compress(VectorOperation::add);

      std::cout << "after: \n";
      gpu_vec.print(std::cout);
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
