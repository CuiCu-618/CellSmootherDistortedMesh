/**
 * @file cuda_mg_transfer.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief Implementation of the grid transfer operations.
 * @version 1.0
 * @date 2023-02-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef MG_TRANSFER_CUH
#define MG_TRANSFER_CUH

#include <deal.II/base/config.h>

#include <deal.II/base/logstream.h>
#include <deal.II/base/mg_level_object.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/shape_info.h>

#include <deal.II/multigrid/mg_base.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_transfer_internal.h>

#include "cuda_vector.cuh"
#include "utilities.cuh"

using namespace dealii;

namespace PSMF
{

  struct IndexMapping
  {
    CudaVector<unsigned int> global_indices;
    CudaVector<unsigned int> level_indices;

    std::size_t
    memory_consumption() const
    {
      return global_indices.memory_consumption() +
             level_indices.memory_consumption();
    }
  };


  /**
   * Implementation of the MGTransferBase interface for which the transfer
   * operations is implemented in a matrix-free way based on the interpolation
   * matrices of the underlying finite element. This requires considerably
   * less memory than MGTransferPrebuilt and can also be considerably faster
   * than that variant.
   */
  template <int dim, typename Number>
  class MGTransferCUDA
    : public MGTransferBase<LinearAlgebra::distributed::Vector<
        Number,
        MemorySpace::CUDA>> // public Subscriptor
  {
  public:
    /**
     * Constructor without constraint. Use this constructor only with
     * discontinuous finite elements or with no local refinement.
     */
    MGTransferCUDA();

    /**
     * Constructor with constraints. Equivalent to the default constructor
     * followed by initialize_constraints().
     */
    MGTransferCUDA(const MGConstrainedDoFs &mg_constrained_dofs);

    /**
     * Destructor.
     */
    ~MGTransferCUDA();

    /**
     * Initialize the constraints to be used in build().
     */
    void
    initialize_constraints(const MGConstrainedDoFs &mg_constrained_dofs);

    /**
     * Reset the object to the state it had right after the default
     * constructor.
     */
    void
    clear();

    /**
     * Actually build the information for the prolongation for each level.
     */
    void
    build(
      const DoFHandler<dim, dim> &mg_dof,
      const std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>
        &external_partitioners =
          std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>());

    /**
     * Prolongate a vector from level <tt>to_level-1</tt> to level
     * <tt>to_level</tt> using the embedding matrices of the underlying finite
     * element. The previous content of <tt>dst</tt> is overwritten.
     *
     * @param src is a vector with as many elements as there are degrees of
     * freedom on the coarser level involved.
     *
     * @param dst has as many elements as there are degrees of freedom on the
     * finer level.
     */
    void
    prolongate(
      const unsigned int                                             to_level,
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src)
      const;

    /**
     * Prolongate a vector from level <tt>to_level-1</tt> to level
     * <tt>to_level</tt> using the embedding matrices of the underlying finite
     * element, summing into the previous content of <tt>dst</tt>.
     *
     * @param src is a vector with as many elements as there are degrees of
     * freedom on the coarser level involved.
     *
     * @param dst has as many elements as there are degrees of freedom on the
     * finer level.
     */
    void
    prolongate_and_add(
      const unsigned int                                             to_level,
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src)
      const;

    /**
     * Restrict a vector from level <tt>from_level</tt> to level
     * <tt>from_level-1</tt> using the transpose operation of the prolongate()
     * method. If the region covered by cells on level <tt>from_level</tt> is
     * smaller than that of level <tt>from_level-1</tt> (local refinement),
     * then some degrees of freedom in <tt>dst</tt> are active and will not be
     * altered. For the other degrees of freedom, the result of the
     * restriction is added.
     *
     * @param src is a vector with as many elements as there are degrees of
     * freedom on the finer level involved.
     *
     * @param dst has as many elements as there are degrees of freedom on the
     * coarser level.
     */
    void
    restrict_and_add(
      const unsigned int                                             from_level,
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src)
      const;

    /**
     * Transfer from multi-level vector to normal vector.
     *
     * Copies data from active portions of an MGVector into the respective
     * positions of a <tt>Vector<number></tt>. In order to keep the result
     * consistent, constrained degrees of freedom are set to zero.
     */
    template <int spacedim, typename Number2>
    void
    copy_to_mg(
      const DoFHandler<dim, spacedim> &mg_dof,
      MGLevelObject<
        LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>>     &dst,
      const LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA> &src)
      const;

    /**
     * Transfer from multi-level vector to normal vector.
     *
     * Copies data from active portions of an MGVector into the respective
     * positions of a <tt>Vector<number></tt>. In order to keep the result
     * consistent, constrained degrees of freedom are set to zero.
     */
    template <int spacedim, typename Number2>
    void
    copy_from_mg(
      const DoFHandler<dim, spacedim>                                &mg_dof,
      LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA> &dst,
      const MGLevelObject<
        LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>> &src)
      const;

    /**
     * Add a multi-level vector to a normal vector.
     *
     * Works as the previous function, but probably not for continuous
     * elements.
     */
    template <int spacedim, typename Number2>
    void
    copy_from_mg_add(
      const DoFHandler<dim, spacedim>                                &mg_dof,
      LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA> &dst,
      const MGLevelObject<
        LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>> &src)
      const;

    /**
     * Memory used by this object.
     */
    std::size_t
    memory_consumption() const;

  private:
    /**
     * A variable storing the degree of the finite element contained in the
     * DoFHandler passed to build(). The selection of the computational kernel
     * is based on this number.
     */
    unsigned int fe_degree;

    /**
     * A variable storing whether the element is continuous and there is a
     * joint degree of freedom in the center of the 1D line.
     */
    bool element_is_continuous;

    /**
     * A variable storing the number of components in the finite element
     * contained in the DoFHandler passed to build().
     */
    unsigned int n_components;

    /**
     * A variable storing the number of degrees of freedom on all child cells.
     * It is <tt>2<sup>dim</sup>*fe.dofs_per_cell</tt> for DG elements and
     * somewhat less for continuous elements.
     */
    unsigned int n_child_cell_dofs;

    /**
     * This variable holds the indices for cells on a given level, extracted
     * from DoFHandler for fast access. All DoF indices on a given level are
     * stored as a plain array (since this class assumes constant DoFs per
     * cell). To index into this array, use the cell number times
     * dofs_per_cell.
     *
     * This array first is arranged such that all locally owned level cells
     * come first (found in the variable n_owned_level_cells) and then other
     * cells necessary for the transfer to the next level.
     *
     */
    std::vector<CudaVector<unsigned int>> level_dof_indices;

    /**
     * A variable storing the connectivity from parent to child cell numbers
     * for each level.
     */
    std::vector<CudaVector<unsigned int>> child_offset_in_parent;

    /**
     * A variable storing the number of cells owned on a given process (sets
     * the bounds for the worker loops) for each level.
     */
    std::vector<unsigned int> n_owned_level_cells;

    /**
     * Holds the one-dimensional embedding (prolongation) matrix from mother
     * element to the children.
     */
    LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
      prolongation_matrix_1d;

    /**
     * For continuous elements, restriction is not additive and we need to
     * weight the result at the end of prolongation (and at the start of
     * restriction) by the valence of the degrees of freedom, i.e., on how
     * many elements they appear. We store the data in vectorized form to
     * allow for cheap access. Moreover, we utilize the fact that we only need
     * to store <tt>3<sup>dim</sup></tt> indices.
     *
     * Data is organized in terms of each level (outer vector) and the cells
     * on each level (inner vector).
     */
    std::vector<LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>>
      weights_on_refined;

    /**
     * Mapping for the copy_to_mg() and copy_from_mg() functions. Here only
     * index pairs locally owned is stored.
     * The data is organized as follows: one table per level. This table has
     * two rows. The first row contains the global index, the second one the
     * level index.
     */
    std::vector<IndexMapping> copy_indices;

    /**
     * This variable stores whether the copy operation from the global to the
     * level vector is actually a plain copy to the finest level. This means
     * that the grid has no adaptive refinement and the numbering on the
     * finest multigrid level is the same as in the global case.
     */
    bool perform_plain_copy;

    /**
     * A variable storing the local indices of Dirichlet boundary conditions
     * on cells for all levels (outer index), the cells within the levels
     * (second index), and the indices on the cell (inner index).
     */
    std::vector<CudaVector<unsigned int>> dirichlet_indices;

    /**
     * A vector that holds shared pointers to the partitioners of the
     * transfer. These partitioners might be shared with what was passed in
     * from the outside through build() or be shared with the level vectors
     * inherited from MGLevelGlobalTransfer.
     */
    MGLevelObject<std::shared_ptr<const Utilities::MPI::Partitioner>>
      vector_partitioners;

    /**
     * The mg_constrained_dofs of the level systems.
     */
    SmartPointer<const MGConstrainedDoFs, MGTransferCUDA<dim, Number>>
      mg_constrained_dofs;

    /**
     * Internal function to fill copy_indice.
     */
    void
    fill_copy_indices(const DoFHandler<dim> &mg_dof);

    /**
     * Internal function to transfer data.
     */
    template <typename VectorType, typename VectorType2>
    void
    copy_to_device(VectorType &device, const VectorType2 &host);

    /**
     * Perform the loop_body(prolongation/restriction) operation.
     */
    template <template <int, int, typename> class loop_body, int degree>
    void
    coarse_cell_loop(
      const unsigned int                                             fine_level,
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src)
      const;

    void
    set_mg_constrained_dofs(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &vec,
      unsigned int                                                   level,
      Number                                                         val) const;
  };

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


} // namespace PSMF


// #include "cuda_mg_transfer.template.cuh"

/**
 * \page cuda_mg_transfer
 * \include cuda_mg_transfer.cuh
 */

#endif // MG_TRANSFER_CUH