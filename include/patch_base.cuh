/**
 * @file patch_base.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief This class collects all the data that is stored for the matrix free implementation.
 * @version 1.0
 * @date 2023-02-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef PATCH_BASE_CUH
#define PATCH_BASE_CUH

#include <deal.II/grid/filtered_iterator.h>

#include "tensor_product.h"
#include "utilities.cuh"

using namespace dealii;

/**
 * Namespace for the Patch Smoother Matrix-Free
 */
namespace PSMF
{

  /**
   * @brief Laplace Variant: kernel type for Laplace operator.
   */
  enum class LaplaceVariant
  {
    /**
     * Basic implementation.
     */
    Basic,

    /**
     * Basic implementation. Load data cell by cell.
     */
    BasicCell,

    /**
     * A conflict-free implementation by restructuring shared memory access.
     */
    ConflictFree,

    /**
     * Using the Warp Matrix Multiply and Accumulate (WMMA) API introduced in
     * CUDA 11.0.
     */
    TensorCore,

    /**
     * Using the Matrix Multiply and Accumulate ISA with inline PTX.
     */
    TensorCoreMMA
  };


  /**
   * @brief Smoother Variant: kernel type for
   * Multiplicative Schwarz Smoother.
   */
  enum class SmootherVariant
  {
    /**
     * Compute the residual globally, i.e.
     * r = b - Ax, where A is the system matrix.
     */
    GLOBAL,

    /**
     * Same as above, but with linear thread indicex instead of tmasking
     * boundary threads for local solver.
     */
    FUSED_L,

    /**
     * A conflict-free implementation by restructuring shared memory access.
     */
    ConflictFree,

    /**
     * Using the Warp Matrix Multiply and Accumulate (WMMA) API introduced in
     * CUDA 11.0.
     */
    TensorCore,

    /**
     * Compute the residual b - Ax exactly with FE_DGQHermite element.
     */
    ExactRes
  };


  enum class DoFLayout
  {
    DGQ,
    Q,
    RT
  };

  enum class SolverVariant
  {
    GMRES,
    PCG,
    FMG,
    Linear_FMG,
    Vcycle
  };

  enum class ExpementsSets
  {
    none,
    kernel,
    error_analysis,
    solvers,
    vnum
  };

  /**
   * Granularity Scheme: number of patches per thread-block
   */
  enum class GranularityScheme
  {
    none,
    user_define,
    multiple
  };


  /**
   * @brief Implementation for Discontinuous Galerkin(DG) element
   *
   * @tparam dim
   * @tparam fe_degree
   * @tparam Number
   */
  template <int dim, int fe_degree, typename Number>
  class LevelVertexPatch : public Subscriptor
  {
  public:
    using CellIterator = typename DoFHandler<dim>::level_cell_iterator;
    using PatchIterator =
      typename std::vector<std::vector<CellIterator>>::const_iterator;

    static constexpr unsigned int regular_vpatch_size = 1 << dim;
    static constexpr unsigned int n_patch_dofs =
      Util::pow(2 * fe_degree + 2, dim);

    /**
     * Standardized data struct to pipe additional data to LevelVertexPatch.
     */
    struct AdditionalData
    {
      /**
       * Constructor.
       */
      AdditionalData(
        const Number            relaxation         = 1.,
        const bool              use_coloring       = true,
        const unsigned int      patch_per_block    = 1,
        const GranularityScheme granularity_scheme = GranularityScheme::none)
        : relaxation(relaxation)
        , use_coloring(use_coloring)
        , patch_per_block(patch_per_block)
        , granularity_scheme(granularity_scheme)
      {}

      /**
       * Relaxation parameter.
       */
      Number relaxation;

      /**
       * If true, use coloring. Otherwise, use atomic operations.
       * Coloring ensures bitwise reproducibility but is slower on Pascal and
       * newer architectures (need to check).
       */
      bool use_coloring;

      /**
       * Number of patches per thread block.
       */
      unsigned int patch_per_block;

      GranularityScheme granularity_scheme;
    };


    /**
     * Structure which is passed to the kernel.
     * It is used to pass all the necessary information from the CPU to the
     * GPU.
     */
    struct Data
    {
      /**
       * Number of patches for each color.
       */
      unsigned int n_patches;

      /**
       * Number of patches per thread block.
       */
      unsigned int patch_per_block;

      /**
       * Number of ghost indices
       */
      unsigned int n_ghost_indices;

      /**
       * The range of the vector that is stored locally.
       */
      unsigned int local_range_start;
      unsigned int local_range_end;

      /**
       * The set of indices to which we need to have read access but that are
       * not locally owned.
       */
      unsigned int *ghost_indices;

      /**
       * Return the local index corresponding to the given global index.
       */
      __device__ unsigned int
      global_to_local(const unsigned int global_index) const;

      __device__ unsigned int
      binary_search(const unsigned int local_index,
                    const unsigned int l,
                    const unsigned int r) const;

      __device__ bool
      is_ghost(const unsigned int global_index) const;

      /**
       * Relaxation parameter.
       */
      Number relaxation;

      /**
       * Pointer to the the first degree of freedom in each patch.
       * @note Need Lexicographic ordering degree of freedoms.
       * @note For DG case, the first degree of freedom index of
       *       four cells in a patch is stored consecutively.
       */
      unsigned int *first_dof;
      unsigned int *patch_dofs;

      /**
       * Pointer to the patch cell ordering type.
       */
      unsigned int *patch_id;

      /**
       * Pointer to the patch type. left, middle, right
       */
      unsigned int *patch_type;

      /**
       * Pointer to mapping from l to h
       */
      unsigned int *l_to_h;

      /**
       * Pointer to mapping from l to h
       */
      unsigned int *h_to_l;

      /**
       * Pointer to 1D mass matrix for lapalace operator.
       */
      Number *laplace_mass_1d;

      /**
       * Pointer to 1D stiffness matrix for lapalace operator.
       */
      Number *laplace_stiff_1d;

      /**
       * Pointer to 1D mass matrix for smoothing operator.
       */
      Number *smooth_mass_1d;

      /**
       * Pointer to 1D stiffness matrix for smoothing operator.
       */
      Number *smooth_stiff_1d;

      /**
       * Pointer to 1D eigenvalues for smoothing operator.
       */
      Number *eigenvalues;

      /**
       * Pointer to 1D eigenvectors for smoothing operator.
       */
      Number *eigenvectors;
    };

    struct GhostPatch
    {
      GhostPatch(const unsigned int proc, const CellId &cell_id);

      void
      submit_id(const unsigned int proc, const CellId &cell_id);

      std::string
      str() const;

      std::map<unsigned, std::vector<CellId>> proc_to_cell_ids;
    };

    /**
     * Default constructor.
     */
    LevelVertexPatch();

    /**
     * Destructor.
     */
    ~LevelVertexPatch();

    /**
     * Return the Data structure associated with @p color for lapalce operator.
     */
    Data
    get_laplace_data(unsigned int color) const;

    Data
    get_laplace_data_ghost(unsigned int color) const;

    /**
     * Return the Data structure associated with @p color for smoothing operator.
     */
    Data
    get_smooth_data(unsigned int color) const;

    Data
    get_smooth_data_ghost(unsigned int color) const;

    /**
     * Extracts the information needed to perform loops over cells.
     */
    void
    reinit(const DoFHandler<dim> &dof_handler,
           const unsigned int     mg_level,
           const AdditionalData  &additional_data = AdditionalData());

    /**
     * @brief This method runs the loop over all patches and apply the local operation on
     * each element in parallel.
     *
     * @tparam Operator a operator which is applied on each patch
     * @tparam VectorType
     * @param func
     * @param src
     * @param dst
     */
    template <typename Operator, typename VectorType>
    void
    patch_loop(const Operator   &op,
               const VectorType &src,
               VectorType       &dst) const;

    /**
     * This method runs the loop over all cells and compute tensor product on
     * each element in parallel for global mat-vec operation.
     * We call it 'cell loop' because although patches are used as a unit, only
     * some of the cells are updated, thus avoiding the duplication of face
     * integrals.
     */
    template <typename Operator, typename VectorType>
    void
    cell_loop(const Operator &op, const VectorType &src, VectorType &dst) const;

    /**
     * @brief Initializes the tensor product matrix for local smoothing.
     */
    void
    reinit_tensor_product_smoother() const;

    /**
     * @brief Initializes the tensor product matrix for local laplace.
     */
    void
    reinit_tensor_product_laplace() const;

    /**
     * Free all the memory allocated.
     */
    void
    free();

    /**
     * Return an approximation of the memory consumption of this class in
     * bytes.
     */
    std::size_t
    memory_consumption() const;

  private:
    /**
     * Helper function. Setup color arrays for collecting data.
     */
    void
    setup_color_arrays(const unsigned int n_colors);

    /**
     * Helper function. Setup color arrays for collecting data.
     */
    void
    setup_configuration(const unsigned int n_colors);

    /**
     * Helper function. Get tensor product data for each patch.
     */
    void
    get_patch_data(const PatchIterator &patch,
                   const unsigned int   patch_id,
                   const bool           is_ghost = false);

    /**
     * Gathering the locally owned and ghost cells attached to a common
     * vertex as the collection of cell iterators (patch). The
     * successive distribution of the collections with ghost cells
     * follows the logic:
     *
     * 1.) if one mpi-proc owns more than half of the cells (locally owned)
     * of the vertex patch the mpi-proc takes ownership
     *
     * 2.) for the remaining ghost patches the mpi-proc with the cell of
     * the lowest CellId (see dealii::Triangulation) takes ownership
     */
    std::vector<std::vector<CellIterator>>
    gather_vertex_patches(const DoFHandler<dim> &dof_handler,
                          const unsigned int     level) const;

    /**
     * Allocate an array to the device.
     */
    template <typename Number1>
    void
    alloc_arrays(Number1 **array_device, const unsigned int n);

    /**
     * Number of global refinments.
     */
    unsigned int level;

    /**
     * Number of colors produced by the coloring algorithm.
     */
    unsigned int n_colors;

    /**
     * Relaxation parameter.
     */
    Number relaxation;

    /**
     * If true, use coloring. Otherwise, use atomic operations.
     * Coloring ensures bitwise reproducibility but is slower on Pascal and
     * newer architectures.
     */
    bool use_coloring;

    /**
     * Number of coarse cells
     */
    unsigned int n_replicate;


    GranularityScheme granularity_scheme;

    /**
     * Grid dimensions associated to the different colors. The grid dimensions
     * are used to launch the CUDA kernels.
     */
    std::vector<dim3> grid_dim_lapalce;
    std::vector<dim3> grid_dim_smooth;

    std::vector<dim3> grid_dim_lapalce_ghost;
    std::vector<dim3> grid_dim_smooth_ghost;

    /**
     * Block dimensions associated to the different colors. The block
     * dimensions are used to launch the CUDA kernels.
     */
    std::vector<dim3> block_dim_laplace;
    std::vector<dim3> block_dim_smooth;

    /**
     * Number of patches per thread block.
     */
    unsigned int patch_per_block;

    /**
     * Auxiliary vector.
     */
    mutable LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> tmp;

    /**
     * Raw graphed of locally owned active patches.
     */
    std::vector<std::vector<PatchIterator>> graph_ptr_raw;
    std::vector<std::vector<PatchIterator>> graph_ptr_raw_ghost;

    /**
     * Colored graphed of locally owned active patches.
     */
    std::vector<std::vector<PatchIterator>> graph_ptr_colored;
    std::vector<std::vector<PatchIterator>> graph_ptr_colored_ghost;

    /**
     * Number of patches in each color.
     */
    std::vector<unsigned int> n_patches_laplace;
    std::vector<unsigned int> n_patches_smooth;

    std::vector<unsigned int> n_patches_laplace_ghost;
    std::vector<unsigned int> n_patches_smooth_ghost;

    /**
     * Pointer to the DoFHandler associated with the object.
     */
    const DoFHandler<dim> *dof_handler;

    /**
     * Vector of pointer to the the first degree of freedom
     * in each patch of each color.
     * @note Need Lexicographic ordering degree of freedoms.
     * @note For DG case, the first degree of freedom index of
     *       four cells in a patch is stored consecutively.
     */
    std::vector<unsigned int *> first_dof_laplace;
    std::vector<unsigned int *> first_dof_smooth;

    std::vector<unsigned int *> patch_dofs_laplace;
    std::vector<unsigned int *> patch_dofs_smooth;

    /**
     * Vector of the the first degree of freedom
     * in each patch of a single color.
     * Initialize on host and copy to device later.
     */
    std::vector<unsigned int> first_dof_host;
    std::vector<unsigned int> patch_dofs_host;

    /**
     * Vector of pointer to patch type: left, middle, right.
     */
    std::vector<unsigned int *> patch_type;
    std::vector<unsigned int *> patch_type_ghost;

    /**
     * Vector of patch type: left, middle, right.
     * Initialize on host and copy to device later.
     */
    std::vector<unsigned int> patch_type_host;

    /**
     * Vector of pointer to the local cell ordering type for each patch.
     */
    std::vector<unsigned int *> patch_id;

    /**
     * Vector of patch cell ordering type.
     * Initialize on host and copy to device later.
     */
    std::vector<unsigned int> patch_id_host;

    /**
     * Mapping from cell ordering to type id.
     */
    std::unordered_map<unsigned int, unsigned int> ordering_to_type;

    /**
     * Categories of ordering types.
     */
    unsigned int ordering_types;

    /**
     * Lookup table
     */
    std::unordered_map<unsigned int, std::array<unsigned int, 3>> lookup_table;

    /**
     * Pointer to mapping from l to h
     */
    unsigned int *l_to_h;

    std::vector<unsigned int> l_to_h_host;

    /**
     * Pointer to mapping from l to h
     */
    unsigned int *h_to_l;

    std::vector<unsigned int> h_to_l_host;

    /**
     * Pointer to 1D mass matrix for lapalace operator.
     */
    Number *laplace_mass_1d;

    /**
     * Pointer to 1D stiffness matrix for lapalace operator.
     */
    Number *laplace_stiff_1d;

    /**
     * Pointer to 1D mass matrix for smoothing operator.
     */
    Number *smooth_mass_1d;

    /**
     * Pointer to 1D stiffness matrix for smoothing operator.
     */
    Number *smooth_stiff_1d;

    /**
     * Pointer to 1D eigenvalues for smoothing operator.
     */
    Number *eigenvalues;

    /**
     * Pointer to 1D eigenvectors for smoothing operator.
     */
    Number *eigenvectors;

    /**
     * Number of ghost indices
     */
    unsigned int n_ghost_indices;

    /**
     * The range of the vector that is stored locally.
     */
    unsigned int local_range_start;
    unsigned int local_range_end;

    /**
     * The set of indices to which we need to have read access but that are
     * not locally owned.
     */
    unsigned int *ghost_indices_dev;

    /**
     * Shared pointer to store the parallel partitioning information. This
     * information can be shared between several vectors that have the same
     * partitioning.
     */
    std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;

    mutable std::shared_ptr<
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>>
      solution_ghosted;

    cudaStream_t stream;
    cudaStream_t stream_g;
  };

  /**
   * Structure to pass the shared memory into a general user function.
   * TODO: specialize for cell loop and patch loop
   */
  template <int dim, typename Number, bool is_laplace>
  struct SharedMemData
  {
    /**
     * Constructor.
     */
    __device__
    SharedMemData(Number      *data,
                  unsigned int n_buff,
                  unsigned int n_dofs_1d,
                  unsigned int local_dim,
                  unsigned int n_dofs_1d_padding = 0)
    {
      constexpr unsigned int n = is_laplace ? 3 : 1;
      n_dofs_1d_padding =
        n_dofs_1d_padding == 0 ? n_dofs_1d : n_dofs_1d_padding;

      local_src = data;
      local_dst = local_src + n_buff * local_dim;

      local_mass = local_dst + n_buff * local_dim;
      local_derivative =
        local_mass + n_buff * n_dofs_1d * n_dofs_1d_padding * n;
      tmp = local_derivative + n_buff * n_dofs_1d * n_dofs_1d_padding * n;
    }


    /**
     * Shared memory for local and interior src.
     */
    Number *local_src;

    /**
     * Shared memory for local and interior dst.
     */
    Number *local_dst;

    /**
     * Shared memory for local and interior residual.
     */
    Number *local_residual;

    /**
     * Shared memory for computed 1D mass matrix.
     */
    Number *local_mass;

    /**
     * Shared memory for computed 1D Laplace matrix.
     */
    Number *local_derivative;

    /**
     * Shared memory for computed 1D eigenvalues.
     */
    Number *local_eigenvalues;

    /**
     * Shared memory for computed 1D eigenvectors.
     */
    Number *local_eigenvectors;

    /**
     * Shared memory for internal buffer.
     */
    Number *tmp;
  };

  /**
   * This function determines number of patches per block at compile time.
   */
  template <int dim, int fe_degree>
  __host__ __device__ constexpr unsigned int
  granularity_shmem()
  {
    return dim == 2 ? (fe_degree == 1 ? 32 :
                       fe_degree == 2 ? 16 :
                       fe_degree == 3 ? 4 :
                       fe_degree == 4 ? 2 :
                                        1) :
           dim == 3 ? (fe_degree == 1 ? 16 :
                       fe_degree == 2 ? 2 :
                                        1) :
                      1;
  }

  /**
   * Renumbering indices for vectors of tensor-dimension 2 so that wmma can be
   * applied.
   */
  __constant__ unsigned int numbering2[8 * 8 * 8];

} // namespace PSMF

#include "patch_base.template.cuh"

/**
 * \page patch_base
 * \include patch_base.cuh
 */

#endif // PATCH_BASE_CUH