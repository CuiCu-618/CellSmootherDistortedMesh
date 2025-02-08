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

#include <execution>
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


  constexpr unsigned int padding   = 0;
  constexpr unsigned int BLOCK_DIM = 256;

  constexpr unsigned int DIM        = 3;
  constexpr unsigned int MAX_DEGREE = 4;
  constexpr unsigned int MAX_CELL_DOFS_RT =
    DIM * pow(MAX_DEGREE + 1, DIM - 1) * (MAX_DEGREE + 2);
  constexpr unsigned int MAX_PATCH_DOFS_RT =
    DIM * pow(2 * MAX_DEGREE + 2, DIM - 1) * (2 * MAX_DEGREE + 3);
  constexpr unsigned int MAX_PATCH_DOFS_RT_INT =
    DIM * pow(2 * MAX_DEGREE + 2, DIM - 1) * (2 * MAX_DEGREE + 1);
  constexpr unsigned int MAX_PATCH_DOFS_DG = pow(2 * MAX_DEGREE + 2, DIM);

} // namespace Util

#endif // UTILITIES_CUH