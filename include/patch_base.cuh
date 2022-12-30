/**
 * @file patch_base.cuh
 * Created by Cu Cui on 2022/12/25.
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
   * Smoother Variant: kernel type for
   * Multiplicative Schwarz Smoother.
   */
  enum class SmootherVariant
  {
    FUSED,
    SEPERATE,
    GLOBAL
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


  template <int dim,
            int fe_degree,
            typename Number,
            SmootherVariant kernel,
            DoFLayout       dof_layout>
  class LevelVertexPatch;

  /**
   * @brief Implementation for Continuous Galerkin(CG) element
   *
   * @tparam dim
   * @tparam fe_degree
   * @tparam Number
   * @tparam kernel kernel type
   */
  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  class LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::Q>
    : public Subscriptor
  {
  public:
    using CellFilter =
      FilteredIterator<typename DoFHandler<dim>::level_cell_iterator>;

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
        const unsigned int      patch_per_block    = 1,
        const GranularityScheme granularity_scheme = GranularityScheme::none)
        : relaxation(relaxation)
        , patch_per_block(patch_per_block)
        , granularity_scheme(granularity_scheme)
      {}

      /**
       * Relaxation parameter.
       */
      Number relaxation;

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
       * Relaxation parameter.
       */
      Number relaxation;

      /**
       * Pointer to the the first degree of freedom in each patch.
       * @note Lexicographic ordering is needed.
       */
      unsigned int *first_dof;

      /**
       * Pointer to the patch id.
       * @note Lexicographic ordering is needed.
       */
      unsigned int *patch_id;

      /**
       * Pointer to 1D global mass matrix.
       */
      Number *global_mass_1d;

      /**
       * Pointer to 1D global derivative matrix.
       */
      Number *global_derivative_1d;

      /**
       * Pointer to 1D eigenvalues.
       */
      Number *eigenvalues;

      /**
       * Pointer to 1D eigenvectors.
       */
      Number *eigenvectors;
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
     * Return the Data structure associated with @p color.
     */
    Data
    get_data(unsigned int color) const;

    /**
     * Extracts the information needed to perform loops over cells.
     */
    template <typename MatrixFreeType>
    void
    reinit(const MatrixFreeType &matrix_free,
           const AdditionalData &additional_data = AdditionalData());

    /**
     * @brief This method runs the loop over all patches and apply the local operation on
     * each element in parallel.
     *
     * @tparam Functor a functor which is applied on each patch
     * @tparam VectorType
     * @tparam Functor_inv a functor which is applied on each patch
     * @param func
     * @param src
     * @param dst
     * @param func_inv
     */
    template <typename Functor,
              typename VectorType,
              typename Functor_inv = Functor>
    void
    patch_loop(const Functor     &func,
               const VectorType  &src,
               VectorType        &dst,
               const Functor_inv &func_inv = Functor_inv()) const;

    /**
     * Helper function. Loop over all the patches and apply the functor on
     * each element in parallel. GLOBAL kernel.
     */
    template <typename MatrixType, typename Functor_inv, typename VectorType>
    void
    patch_loop_global(const MatrixType  &A,
                      const Functor_inv &func_inv,
                      const VectorType  &src,
                      VectorType        &dst) const;

    /**
     * @brief Initializes the tensor product matrix.
     */
    void
    reinit_tensor_product() const;

    /**
     * Return the remaining patches after coloring.
     */
    const std::set<unsigned int>
    get_colored_graph_left() const;

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
     * Helper function. Loop over all the patches and apply the functor on
     * each element in parallel. FUSED kernel.
     */
    template <typename Functor, typename VectorType>
    void
    patch_loop_fused(const Functor    &func,
                     const VectorType &src,
                     VectorType       &dst) const;

    /**
     * Helper function. Loop over all the patches and apply the functor on
     * each element in parallel. SEPERATE kernel.
     */
    template <typename Functor, typename Functor_inv, typename VectorType>
    void
    patch_loop_seperate(const Functor     &func,
                        const Functor_inv &func_inv,
                        const VectorType  &src,
                        VectorType        &dst) const;

    /**
     * Helper function. Implement multiple red-black coloring.
     */
    void
    rb_coloring();

    /**
     * Helper function. Setup color arrays for collecting data.
     */

    void
    setup_color_arrays(const unsigned int n_colors);

    /**
     * Helper function. Setup patch arrays for each color.
     */
    void
    setup_patch_arrays(const unsigned int color);

    /**
     * Helper function. Get tensor product data for each patch.
     */
    void
    get_patch_data(const unsigned int patch);

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


    GranularityScheme granularity_scheme;

    /**
     * Grid dimensions associated to the different colors. The grid dimensions
     * are used to launch the CUDA kernels.
     */
    std::vector<dim3> grid_dim;

    /**
     * Block dimensions associated to the different colors. The block
     * dimensions are used to launch the CUDA kernels.
     */
    std::vector<dim3> block_dim;
    std::vector<dim3> block_dim_inv;

    /**
     * Number of patches per thread block.
     */
    unsigned int patch_per_block;

    /**
     * Auxiliary vector.
     */
    mutable LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> tmp;


    /**
     * Colored graphed of locally owned active patches.
     */
    std::vector<std::vector<unsigned int>> graph;

    /**
     * Colored graphed of locally owned active patches.
     */
    std::vector<std::vector<CellFilter>> graph_ptr;

    /**
     * Number of patches in each color.
     */
    std::vector<unsigned int> n_patches;

    /**
     * Pointer to the DoFHandler associated with the object.
     */
    const DoFHandler<dim> *dof_handler;

    /**
     * Vector of pointer to the the first degree of freedom in each patch of
     * each color.
     * @note Need Lexicographic ordering degree of freedoms.
     */
    std::vector<unsigned int *> first_dof;

    /**
     * Vector of pointer to the patch id of each color.
     * @note Need Lexicographic ordering patches.
     */
    std::vector<unsigned int *> patch_id;

    /**
     * Pointer to 1D global mass matrix.
     */
    Number *global_mass_1d;

    /**
     * Pointer to 1D global derivative matrix.
     */
    Number *global_derivative_1d;

    /**
     * Vector of pointer to 1D eigenvalues of each color.
     */
    Number *eigenvalues;

    /**
     * Vector of pointer to 1D eigenvectors of each color.
     */
    Number *eigenvectors;
  };

  /**
   * Structure to pass the shared memory into a general user function.
   */
  template <int dim, typename Number, SmootherVariant kernel>
  struct SharedMemData
  {
    /**
     * Constructor.
     */
    __device__
    SharedMemData(Number      *data,
                  unsigned int n_buff,
                  unsigned int n_dofs_1d,
                  unsigned int local_dim)
    {
      local_src = data;
      local_dst = local_src + n_buff * local_dim;

      local_mass       = local_dst + n_buff * local_dim;
      local_derivative = local_mass + 1 * n_dofs_1d * n_dofs_1d * 1;
      temp             = local_derivative + 1 * n_dofs_1d * n_dofs_1d * 1;
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
    Number *temp;
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

} // namespace PSMF

#include "patch_base.template.cuh"

/**
 * \page patch_base
 * \include patch_base.cuh
 */

#endif // PATCH_BASE_CUH