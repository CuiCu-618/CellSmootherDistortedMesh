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
                  unsigned int  tid_z = 0)
  {
    const unsigned int z_off = tid_z / (fe_degree + 1);
    const unsigned int y_off = threadIdx.y / (fe_degree + 1);
    const unsigned int x_off = threadIdx.x / (fe_degree + 1) - 2 * local_patch;
    const unsigned int z     = tid_z % (fe_degree + 1);
    const unsigned int y     = threadIdx.y % (fe_degree + 1);
    const unsigned int x     = threadIdx.x % (fe_degree + 1);

    return first_dofs[z_off * 4 + y_off * 2 + x_off] +
           z * (fe_degree + 1) * (fe_degree + 1) + y * (fe_degree + 1) + x;
  }

} // namespace Util

#endif // UTILITIES_CUH