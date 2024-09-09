/**
 * @file cell_base.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief This class collects all the data that is stored for the matrix free implementation.
 * @version 1.0
 * @date 2023-02-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef CELL_BASE_CUH
#define CELL_BASE_CUH

using namespace dealii;

/**
 * Namespace for the Patch Smoother Matrix-Free
 */
namespace PSMF
{

  /**
   * @brief Implementation for Discontinuous Galerkin(DG) element
   *
   * @tparam dim
   * @tparam fe_degree
   * @tparam Number
   */
  template <int dim, int fe_degree, typename Number>
  class LevelCellPatch : public Subscriptor
  {
  public:
    using CellIterator = typename DoFHandler<dim>::level_cell_iterator;

    static constexpr unsigned int n_cell_dofs = Util::pow(fe_degree + 1, dim);

    /**
     * Standardized data struct to pipe additional data to LevelCellPatch.
     */
    struct AdditionalData
    {
      /**
       * Constructor.
       */
      AdditionalData(
        const Number            relaxation         = 1.,
        const unsigned int      cell_per_block     = 1,
        const GranularityScheme granularity_scheme = GranularityScheme::none)
        : relaxation(relaxation)
        , cell_per_block(cell_per_block)
        , granularity_scheme(granularity_scheme)
      {}

      /**
       * Relaxation parameter.
       */
      Number relaxation;

      /**
       * Number of cells per thread block.
       */
      unsigned int cell_per_block;

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
       * Number of cells for each color.
       */
      types::global_dof_index n_cells;

      /**
       * Number of cells per thread block.
       */
      unsigned int cell_per_block;

      /**
       * Number of ghost indices
       */
      types::global_dof_index n_ghost_indices;

      /**
       * The range of the vector that is stored locally.
       */
      types::global_dof_index local_range_start;
      types::global_dof_index local_range_end;

      /**
       * The set of indices to which we need to have read access but that are
       * not locally owned.
       */
      types::global_dof_index *ghost_indices;

      /**
       * Return the local index corresponding to the given global index.
       */
      __device__ types::global_dof_index
      global_to_local(const types::global_dof_index global_index) const;

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
       * Pointer to the the first degree of freedom in each cell.
       * @note Need Lexicographic ordering degree of freedoms.
       * @note For DG case, the first degree of freedom index of
       *       four cells in a cell is stored consecutively.
       */
      types::global_dof_index *first_dof;
      types::global_dof_index *patch_dofs;

      /**
       * Pointer to the cell type. left, middle, right
       */
      unsigned int *cell_type;

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


    /**
     * Default constructor.
     */
    LevelCellPatch();

    /**
     * Destructor.
     */
    ~LevelCellPatch();

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
     * @brief This method runs the loop over all cells and apply the local operation on
     * each element in parallel.
     *
     * @tparam Operator a operator which is applied on each cell
     * @tparam VectorType
     * @param src
     * @param dst
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
     * Helper function. Get tensor product data for each cell.
     */
    void
    get_cell_data(const CellIterator           &cell,
                  const types::global_dof_index cell_id,
                  const bool                    is_ghost = false);

    /**
     * Allocate an array to the device.
     */
    template <typename Number1>
    void
    alloc_arrays(Number1 **array_device, const types::global_dof_index n);

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
     * Number of cells per thread block.
     */
    unsigned int cell_per_block;

    /**
     * Auxiliary vector.
     */
    mutable LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> tmp;

    /**
     * Raw graphed of locally owned active cells.
     */
    std::vector<std::vector<CellIterator>> graph_ptr_raw;
    std::vector<std::vector<CellIterator>> graph_ptr_raw_ghost;

    /**
     * Colored graphed of locally owned active cells.
     */
    std::vector<std::vector<CellIterator>> graph_ptr_colored;
    std::vector<std::vector<CellIterator>> graph_ptr_colored_ghost;

    /**
     * Number of cells in each color.
     */
    std::vector<unsigned int> n_cells_laplace;
    std::vector<unsigned int> n_cells_smooth;

    std::vector<unsigned int> n_cells_laplace_ghost;
    std::vector<unsigned int> n_cells_smooth_ghost;

    /**
     * Pointer to the DoFHandler associated with the object.
     */
    const DoFHandler<dim> *dof_handler;

    /**
     * Vector of pointer to the the first degree of freedom
     * in each cell of each color.
     * @note Need Lexicographic ordering degree of freedoms.
     * @note For DG case, the first degree of freedom index of
     *       four cells in a cell is stored consecutively.
     */
    std::vector<types::global_dof_index *> first_dof_laplace;
    std::vector<types::global_dof_index *> first_dof_smooth;

    std::vector<types::global_dof_index *> patch_dofs_laplace;
    std::vector<types::global_dof_index *> patch_dofs_smooth;

    /**
     * Vector of the the first degree of freedom
     * in each cell of a single color.
     * Initialize on host and copy to device later.
     */
    std::vector<types::global_dof_index> first_dof_host;
    std::vector<types::global_dof_index> cell_dofs_host;

    /**
     * Vector of pointer to cell type: left, middle, right.
     */
    std::vector<unsigned int *> cell_type;

    /**
     * Vector of cell type: left, middle, right.
     * Initialize on host and copy to device later.
     */
    std::vector<unsigned int> cell_type_host;

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
    types::global_dof_index n_ghost_indices;

    /**
     * The range of the vector that is stored locally.
     */
    types::global_dof_index local_range_start;
    types::global_dof_index local_range_end;

    /**
     * The set of indices to which we need to have read access but that are
     * not locally owned.
     */
    types::global_dof_index *ghost_indices_dev;

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
   * TODO: specialize for cell loop and cell loop
   */
  template <int dim, typename Number, bool is_laplace>
  struct CellSharedMemData
  {
    /**
     * Constructor.
     */
    __device__
    CellSharedMemData(Number      *data,
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
      tmp = local_derivative + n_buff * n_dofs_1d * n_dofs_1d_padding * n * dim;
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
   * This function determines number of cells per block at compile time.
   */
  template <int dim, int fe_degree>
  __host__ __device__ constexpr unsigned int
  cell_granularity_shmem()
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
} // namespace PSMF

#include "cell_base.template.cuh"

/**
 * \page cell_base
 * \include cell_base.cuh
 */

#endif // CELL_BASE_CUH
