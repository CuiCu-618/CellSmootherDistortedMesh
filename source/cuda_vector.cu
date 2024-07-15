/**
 * @file cuda_vector.cu
 * Created by Cu Cui on 2022/12/25.
 */

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_parallel_vector.templates.h>
#include <deal.II/lac/read_write_vector.templates.h>

#include "cuda_vector.cuh"
#include "cuda_vector.template.cuh"

namespace PSMF
{

  template class CudaVector<types::global_dof_index>;
  template class CudaVector<float>;
  template class CudaVector<double>;

} // namespace PSMF

DEAL_II_NAMESPACE_OPEN

template class LinearAlgebra::ReadWriteVector<types::global_dof_index>;

namespace LinearAlgebra
{
  namespace distributed
  {
    template void
    Vector<double, ::dealii::MemorySpace::CUDA>::copy_locally_owned_data_from<
      double>(const Vector<double, ::dealii::MemorySpace::CUDA> &);

    template void
    Vector<float, ::dealii::MemorySpace::CUDA>::copy_locally_owned_data_from<
      float>(const Vector<float, ::dealii::MemorySpace::CUDA> &);
  } // namespace distributed
} // namespace LinearAlgebra

DEAL_II_NAMESPACE_CLOSE
