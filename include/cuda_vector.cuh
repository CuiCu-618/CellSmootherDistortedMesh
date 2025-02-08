/**
 * @file cuda_vector.cuh
 * Created by Cu Cui on 2022/12/25.
 */

#ifndef CUDA_VECTOR_CUH
#define CUDA_VECTOR_CUH

#include <deal.II/lac/cuda_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

using namespace dealii;

namespace PSMF
{

  /**
   * Implementation of a cuda vector class.
   * Mainly used for unsigned int.
   */
  template <typename Number = unsigned int>
  class CudaVector : public Subscriptor
  {
  public:
    using value_type      = Number;
    using pointer         = value_type *;
    using const_pointer   = const value_type *;
    using iterator        = value_type *;
    using const_iterator  = const value_type *;
    using reference       = value_type &;
    using const_reference = const value_type &;
    using size_type       = types::global_dof_index;

    CudaVector();

    CudaVector(const CudaVector<Number> &V);

    ~CudaVector();

    void
    reinit(const size_type size, const bool omit_zeroing_entries = false);

    void
    import(const LinearAlgebra::ReadWriteVector<Number> &V,
           VectorOperation::values                       operation);

    Number *
    get_values() const;

    size_type
    size() const;

    std::size_t
    memory_consumption() const;

  private:
    std::unique_ptr<Number[], void (*)(Number *)> val;

    size_type n_elements;
  };

  // ---------------------------- Inline functions --------------------------

  template <typename Number>
  inline Number *
  CudaVector<Number>::get_values() const
  {
    return val.get();
  }

  template <typename Number>
  inline typename CudaVector<Number>::size_type
  CudaVector<Number>::size() const
  {
    return n_elements;
  }


#define BKSIZE_ELEMWISE_OP 512
#define CHUNKSIZE_ELEMWISE_OP 8

  template <bool add, typename Number, typename Number2>
  __global__ void
  vec_equ(Number *dst, const Number2 *src, const int N)
  {
    const int idx_base =
      threadIdx.x + blockIdx.x * (blockDim.x * CHUNKSIZE_ELEMWISE_OP);

    for (int c = 0; c < CHUNKSIZE_ELEMWISE_OP; ++c)
      {
        const int idx = idx_base + c * BKSIZE_ELEMWISE_OP;
        if (idx < N)
          if (add)
            dst[idx] += src[idx];
          else
            dst[idx] = src[idx];
      }
  }

  template <bool add, typename VectorType, typename VectorType2>
  void
  plain_copy(VectorType &dst, const VectorType2 &src)
  {
    if (dst.size() != src.size())
      {
        dst.reinit(src.size(), true);
      }

    const int nblocks =
      1 + (src.size() - 1) / (CHUNKSIZE_ELEMWISE_OP * BKSIZE_ELEMWISE_OP);
    vec_equ<add,
            typename VectorType::value_type,
            typename VectorType2::value_type>
      <<<nblocks, BKSIZE_ELEMWISE_OP>>>(dst.get_values(),
                                        src.get_values(),
                                        src.size());
    AssertCudaKernel();
  }


  template <typename Number, typename Number2>
  __global__ void
  copy_with_indices_kernel(Number             *dst,
                           Number2            *src,
                           const unsigned int *dst_indices,
                           const unsigned int *src_indices,
                           int                 n)
  {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
      {
        dst[dst_indices[i]] = src[src_indices[i]];
      }
    __syncthreads();
  }

  template <typename Number, typename Number2>
  void
  copy_with_indices(
    LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>        &dst,
    const LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA> &src,
    const CudaVector<unsigned int> &dst_indices,
    const CudaVector<unsigned int> &src_indices)
  {
    const int  n         = dst_indices.size();
    const int  blocksize = 256;
    const dim3 block_dim = dim3(blocksize);
    const dim3 grid_dim  = dim3(1 + (n - 1) / blocksize);
    copy_with_indices_kernel<<<grid_dim, block_dim>>>(dst.get_values(),
                                                      src.get_values(),
                                                      dst_indices.get_values(),
                                                      src_indices.get_values(),
                                                      n);
    AssertCudaKernel();
  }

  template <typename Number, typename Number2>
  void
  copy_with_indices(
    LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>        &dst,
    const LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA> &src,
    const unsigned int *dst_indices,
    const unsigned int *src_indices,
    const unsigned int  n)
  {
    if (n == 0)
      return;

    const int  blocksize = 256;
    const dim3 block_dim = dim3(blocksize);
    const dim3 grid_dim  = dim3(1 + (n - 1) / blocksize);
    copy_with_indices_kernel<<<grid_dim, block_dim>>>(
      dst.get_values(), src.get_values(), dst_indices, src_indices, n);
    AssertCudaKernel();
  }

} // namespace PSMF


#endif // CUDA_VECTOR_CUH