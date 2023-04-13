/**
 * @file utilities.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief collection of helper functions
 * @version 1.0
 * @date 2022-12-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef UTILITIES_CUH
#define UTILITIES_CUH

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/hp/fe_values.h>

#include <vector>

using namespace dealii;

namespace Util
{

  constexpr unsigned int padding = 0;

  template <typename T>
  __host__ __device__ constexpr T
  pow(const T base, const int iexp)
  {
    return iexp <= 0 ?
             1 :
             (iexp == 1 ?
                base :
                (((iexp % 2 == 1) ? base : 1) * pow(base * base, iexp / 2)));
  }

  template <typename VectorType>
  void
  adjust_ghost_range_if_necessary(
    const VectorType                                        &vec,
    const std::shared_ptr<const Utilities::MPI::Partitioner> partitioner)
  {
    if (vec.get_partitioner().get() == partitioner.get())
      return;

    VectorType copy_vec(vec);
    const_cast<VectorType &>(vec).reinit(partitioner);
    const_cast<VectorType &>(vec).copy_locally_owned_data_from(copy_vec);
  }

  /**
   * Compute dofs in a patch based on first_dof.
   * Data layout for local vectors:
   * 12 13 | 14 15
   *  8  9 | 10 11
   * ------|------
   *  4  5 |  6  7
   *  0  1 |  2  3
   */
  template <int dim, int fe_degree>
  __device__ unsigned int
  compute_indices(unsigned int *first_dofs,
                  unsigned int  local_patch,
                  unsigned int  local_tid_x,
                  unsigned int  tid_y,
                  unsigned int  tid_z)
  {
    const unsigned int z_off = tid_z / (fe_degree + 1);
    const unsigned int y_off = tid_y / (fe_degree + 1);
    const unsigned int x_off = local_tid_x / (fe_degree + 1);
    const unsigned int z     = tid_z % (fe_degree + 1);
    const unsigned int y     = tid_y % (fe_degree + 1);
    const unsigned int x     = local_tid_x % (fe_degree + 1);

    return first_dofs[z_off * 4 + y_off * 2 + x_off] +
           z * (fe_degree + 1) * (fe_degree + 1) + y * (fe_degree + 1) + x;
  }


  /**
   * Compute dofs in a patch based on first_dof.
   * Data layout for local vectors:
   * 10 11 | 14 15
   *  8  9 | 12 13
   * ------|------
   *  2  3 |  6  7
   *  0  1 |  4  5
   */
  template <int dim, int fe_degree>
  __device__ unsigned int
  compute_indices_cell(unsigned int *first_dofs, unsigned int linear_tid)
  {
    constexpr unsigned int cell_dofs = pow(fe_degree + 1, dim);

    const unsigned int cell           = linear_tid / cell_dofs;
    const unsigned int local_cell_tid = linear_tid % cell_dofs;

    return first_dofs[cell] + local_cell_tid;
  }

  template <int degree>
  inline __device__ unsigned int
  get_permute_base(const unsigned int, const unsigned int)
  {
    return 0;
  }

  template <>
  inline __device__ unsigned int
  get_permute_base<3>(const unsigned int row, const unsigned int z)
  {
    //  If n is a power of 2, (i % n) is equivalent to (i & (n-1));
    unsigned int base1 = (row & 3) < 2 ? 0 : 4;
    unsigned int base2 = (z & 3) * 2;
    return base1 ^ base2;
  }

} // namespace Util

#endif // UTILITIES_CUH