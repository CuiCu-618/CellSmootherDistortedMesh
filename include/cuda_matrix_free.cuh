/**
 * @file cuda_matrix_free.cuh
 * @brief Matrix free class.
 *
 * This class collects all the data that is stored for the matrix free
 * implementation.
 *
 * @author Cu Cui
 * @date 2024-01-18
 * @version 0.1
 *
 * @remark
 * @note
 * @warning
 */


#ifndef CUDA_MATRIX_FREE_CUH
#define CUDA_MATRIX_FREE_CUH

#include <deal.II/base/config.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/cuda_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/hanging_nodes_internal.h>

namespace PSMF
{
  // Forward declaration
  template <int dim, typename Number>
  class ReinitHelper;

  enum MatrixType
  {
    active_matrix,
    level_matrix,
    edge_up_matrix,
    edge_down_matrix,
    interface_matrix,
  };

  /**
   * This class collects all the data that is stored for the matrix free
   * implementation. The storage scheme is tailored towards several loops
   * performed with the same data, i.e., typically doing many matrix-vector
   * products or residual computations on the same mesh.
   *
   * This class does not implement any operations involving finite element basis
   * functions, i.e., regarding the operation performed on the cells. For these
   * operations, the class FEEvaluation is designed to use the data collected in
   * this class.
   *
   * This class implements a loop over all cells (cell_loop()). This loop is
   * scheduled in such a way that cells that share degrees of freedom
   * are not worked on simultaneously, which implies that it is possible to
   * write to vectors in parallel without having to explicitly synchronize
   * access to these vectors and matrices. This class does not implement any
   * shape values, all it does is to cache the respective data. To implement
   * finite element operations, use the class FEEvalutation and class
   * FEFaceEvalutation.
   *
   * This class traverse the cells in a different order than the usual
   * Triangulation class in deal.II.
   *
   * @note Only float and double are supported.
   *
   */
  template <int dim, typename Number = double>
  class MatrixFree : public dealii::Subscriptor
  {
  public:
    using jacobian_type =
      dealii::Tensor<2, dim, dealii::Tensor<1, dim, Number>>;
    using point_type       = dealii::Point<dim, Number>;
    using ActiveCellFilter = dealii::FilteredIterator<
      typename dealii::DoFHandler<dim>::active_cell_iterator>;
    using LevelCellFilter = dealii::FilteredIterator<
      typename dealii::DoFHandler<dim>::level_cell_iterator>;

    /**
     * Standardized data struct to pipe additional data to MatrixFree.
     */
    struct AdditionalData
    {
      /**
       * Constructor.
       */
      AdditionalData(
        const MatrixType          matrix_type = MatrixType::active_matrix,
        const dealii::UpdateFlags mapping_update_flags =
          dealii::update_gradients | dealii::update_JxW_values,
        const dealii::UpdateFlags mapping_update_flags_boundary_faces =
          dealii::update_default,
        const dealii::UpdateFlags mapping_update_flags_inner_faces =
          dealii::update_default,
        const unsigned int mg_level     = dealii::numbers::invalid_unsigned_int,
        const bool         use_coloring = false,
        const unsigned int my_id        = 0,
        const bool         overlap_communication_computation = false)
        : matrix_type(matrix_type)
        , mapping_update_flags(mapping_update_flags)
        , mapping_update_flags_boundary_faces(
            mapping_update_flags_boundary_faces)
        , mapping_update_flags_inner_faces(mapping_update_flags_inner_faces)
        , mg_level(mg_level)
        , use_coloring(use_coloring)
        , my_id(my_id)
        , overlap_communication_computation(overlap_communication_computation)
      {
#ifndef DEAL_II_MPI_WITH_CUDA_SUPPORT
        AssertThrow(
          overlap_communication_computation == false,
          dealii::ExcMessage(
            "Overlapping communication and computation requires CUDA-aware MPI."));
#endif
        if (overlap_communication_computation == true)
          AssertThrow(
            use_coloring == false || overlap_communication_computation == false,
            dealii::ExcMessage(
              "Overlapping communication and coloring are incompatible options. Only one of them can be enabled."));
      }

      /**
       * This flag is used to determine which quantities should be cached. This
       * class can cache data needed for gradient computations (inverse
       * Jacobians), Jacobian determinants (JxW), quadrature points as well as
       * data for Hessians (derivative of Jacobians). By default, only data for
       * gradients and Jacobian determinants times quadrature weights, JxW, are
       * cached. If quadrature points of second derivatives are needed, they
       * must be specified by this field.
       */
      dealii::UpdateFlags mapping_update_flags;

      /**
       * This flag determines the mapping data on boundary faces to be cached.
       * If set to a value different from update_general (default), the face
       * information is explicitly built. Currently, MatrixFree supports to
       * cache the following data on faces: inverse Jacobians, Jacobian
       * determinants (JxW), quadrature points, data for Hessians (derivative of
       * Jacobians), and normal vectors.
       *
       * @note In order to be able to perform a boundary_operation in the
       * MatrixFree::loop(), this field must be set to a value different from
       * UpdateFlags::update_default.
       */
      dealii::UpdateFlags mapping_update_flags_boundary_faces;

      /**
       * This flag determines the mapping data on interior faces to be cached.
       * If set to a value different from update_general (default), the face
       * information is explicitly built. Currently, MatrixFree supports to
       * cache the following data on faces: inverse Jacobians, Jacobian
       * determinants (JxW), quadrature points, data for Hessians (derivative of
       * Jacobians), and normal vectors.
       *
       * @note In order to be able to perform a boundary_operation in the
       * MatrixFree::loop(), this field must be set to a value different from
       * UpdateFlags::update_default.
       */
      dealii::UpdateFlags mapping_update_flags_inner_faces;

      /**
       * This option can be used to define whether we work on a certain level of
       * the mesh, and not the active cells. If set to invalid_unsigned_int
       * (which is the default value), the active cells are gone through,
       * otherwise the level given by this parameter. Note that if you specify
       * to work on a level, its dofs must be distributed by using
       * \texttt{dof_handler.distribute_mg_dofs(fe);}.
       */
      unsigned int mg_level;

      /**
       * If true, use graph coloring. Otherwise, use atomic operations. Graph
       * coloring ensures bitwise reproducibility but is slower on Pascal and
       * newer architectures.
       */
      bool use_coloring;

      /**
       * Unique ID associated with the object.
       */
      int my_id;

      /**
       * Overlap MPI communications with computation. This requires CUDA-aware
       * MPI and use_coloring must be false.
       */
      bool overlap_communication_computation;

      MatrixType matrix_type;
    };

    /**
     * Structure which is passed to the kernel. It is used to pass all the
     * necessary information from the CPU to the GPU.
     */
    struct Data
    {
      /**
       * Pointer to the quadrature points.
       */
      point_type *q_points;

      /**
       * Map the position in the local vector to the position in the global
       * vector.
       */
      dealii::types::global_dof_index *local_to_global;

      /**
       * When using Multigrid Method, for one level coarse, map the position in
       * the local vector to the position in the global vector.
       */
      dealii::types::global_dof_index *l_to_g_coarse;

      /**
       * Map the face to cell id.
       */
      dealii::types::global_dof_index *face2cell_id;

      /**
       * Pointer to the cell Jacobian.
       */
      Number *jacobian;

      /**
       * Pointer to the cell inverse Jacobian.
       */
      Number *inv_jacobian;

      /**
       * Pointer to the cell Jacobian times the weights.
       */
      Number *JxW;

      /**
       * Pointer to the cell quadrature weights.
       */
      Number *q_weights;

      /**
       * Pointer to the determinant cell inverse Jacobian associated to the
       * cells of each color.
       */
      Number *inv_det;

      /**
       * Pointer to the face Jacobian.
       */
      Number *face_jacobian;

      /**
       * Pointer to the face inverse Jacobian.
       */
      Number *face_inv_jacobian;

      /**
       * Pointer to the face Jacobian times the weights.
       */
      Number *face_JxW;

      /**
       * Pointer to the face quadrature weights.
       */
      Number *face_q_weights;

      /**
       * Pointer to the determinant face inverse Jacobian associated to the
       * cells of each color.
       */
      Number *face_inv_det;

      /**
       * Pointer to the unit normal vector on a face.
       */
      Number *normal_vector;

      /**
       * Pointer to the face direction.
       */
      unsigned int *face_number;

      /**
       * Pointer to the subface number.
       */
      int *subface_number;

      /**
       * Pointer to the face orientation.
       */
      int *face_orientation;


      /**
       * ID of the associated MatrixFree object.
       */
      unsigned int id;

      /**
       * Level of the mesh to be worked on.
       */
      unsigned int mg_level;

      /**
       * Whether the entire finite element is primitive
       */
      bool is_primitive;

      /**
       * Number of objects.
       */
      unsigned int n_objs;

      /**
       * Number of cells.
       */
      unsigned int n_cells;

      /**
       * Number of faces.
       */
      unsigned int n_faces;

      /**
       * Length of the padding.
       */
      unsigned int padding_length;

      /**
       * Length of the face padding.
       */
      unsigned int face_padding_length;

      /**
       * Row start (including padding).
       */
      unsigned int row_start;

      /**
       * Mask deciding where constraints are set on a given cell.
       */
      dealii::internal::MatrixFreeFunctions::ConstraintKinds *constraint_mask;

      /**
       * Range of cell constrains.
       */
      dealii::types::global_dof_index *constraint_range;
      dealii::types::global_dof_index *constraint_coarse_range;

      /**
       * If true, use graph coloring has been used and we can simply add into
       * the destingation vector. Otherwise, use atomic operations.
       */
      bool use_coloring;

      /**
       * Pointer to the hanging nodes constrained degrees of freedom.
       */
      dealii::types::global_dof_index *hanging_nodes_constraint;

      dealii::types::global_dof_index *hanging_nodes_constraint_coarse;

      /**
       * Pointer to the hanging nodes constraint indicator.
       */
      int *hanging_nodes_constraint_indicator;

      /**
       * Pointer to the hanging nodes constraint weights.
       */
      Number *hanging_nodes_constraint_weights;

      MatrixType matrix_type;
    };

    /**
     * Default constructor.
     */
    MatrixFree();

    /**
     * Destructor.
     */
    ~MatrixFree();

    /**
     * Return the length of the padding.
     */
    unsigned int
    get_padding_length() const;

    /**
     * Return the length of the face padding.
     */
    unsigned int
    get_face_padding_length() const;

    /**
     * Extracts the information needed to perform loops over cells. The
     * DoFHandler and AffineConstraints objects describe the layout of
     * degrees of freedom, the DoFHandler and the mapping describe the
     * transformation from unit to real cell, and the finite element
     * underlying the DoFHandler together with the quadrature formula
     * describe the local operations. This function takes an IteratorFilters
     * object (predicate) to loop over a subset of the active cells. When using
     * MPI, the predicate should filter out non locally owned cells.
     */
    template <typename IteratorFiltersType>
    void
    reinit(const dealii::Mapping<dim>              &mapping,
           const dealii::DoFHandler<dim>           &dof_handler,
           const dealii::AffineConstraints<Number> &constraints,
           const dealii::Quadrature<1>             &quad,
           const IteratorFiltersType               &iterator_filter,
           const AdditionalData &additional_data = AdditionalData());

    /**
     * Return the Data structure associated with @p color.
     */
    Data
    get_data(unsigned int color) const;

    /**
     * Return the Data structure associated with @p color.
     */
    template <bool is_boundary_face>
    Data
    get_face_data(unsigned int color) const;

    // clang-format off
    /**
     * This method runs the loop over all cells and apply the local operation on
     * each element in parallel. @p cell_operation is a functor which is 
     * applied on each color.
     *
     * @p func needs to define
     * \code
     * __device__ void operator()(
     *   const unsigned int                            cell,
     *   const typename MatrixFree<dim, Number>::Data *gpu_data,
     *   SharedData<dim, Number> *                     shared_data,
     *   const Number *                                src,
     *   Number *                                      dst) const;
     *   static const unsigned int n_dofs_1d;
     *   static const unsigned int n_local_dofs;
     *   static const unsigned int n_q_points;
     * \endcode
     */
    // clang-format on
    template <typename Functor, typename VectorType>
    void
    cell_loop(const Functor    &cell_operation,
              const VectorType &src,
              VectorType       &dst) const;

    /**
     * This method runs the loop over all cells and apply the local operation on
     * each element in parallel. @p cell_operation is a functor which is applied
     * on each color. As opposed to the other variants that only runs a function
     * on cells, this method also takes as arguments a function for the interior
     * faces and for the boundary faces, respectively.
     */
    template <typename Functor, typename VectorType>
    void
    inner_face_loop(const Functor    &face_operation,
                    const VectorType &src,
                    VectorType       &dst) const;

    /**
     * This method runs the loop over all cells and apply the local operation on
     * each element in parallel. @p cell_operation is a functor which is applied
     * on each color. As opposed to the other variants that only runs a function
     * on cells, this method also takes as arguments a function for the interior
     * faces and for the boundary faces, respectively.
     */
    template <typename Functor, typename VectorType>
    void
    boundary_face_loop(const Functor    &face_operation,
                       const VectorType &src,
                       VectorType       &dst) const;

    /**
     * This method runs the loop over all cells and apply the local operation on
     * each element in parallel. This function is very similar to cell_loop()
     * but it uses a simpler functor.
     *
     * @p func needs to define
     * \code
     *  __device__ void operator()(
     *    const unsigned int                            cell,
     *    const typename MatrixFree<dim, Number>::Data *gpu_data);
     * static const unsigned int n_dofs_1d;
     * static const unsigned int n_local_dofs;
     * static const unsigned int n_q_points;
     * \endcode
     */
    template <typename Functor>
    void
    evaluate_coefficients(Functor func) const;

    /**
     * Copy the values of the constrained entries from @p src to @p dst. This is
     * used to impose zero Dirichlet boundary condition.
     */
    void
    copy_constrained_values(
      const dealii::LinearAlgebra::distributed::
        Vector<Number, dealii::MemorySpace::CUDA> &src,
      dealii::LinearAlgebra::distributed::Vector<Number,
                                                 dealii::MemorySpace::CUDA>
        &dst) const;

    /**
     * Set the entries in @p dst corresponding to constrained values to @p val.
     * The main purpose of this function is to set the constrained entries of
     * the source vector used in cell_loop() to zero.
     */
    void
    set_constrained_values(
      const Number val,
      dealii::LinearAlgebra::distributed::Vector<Number,
                                                 dealii::MemorySpace::CUDA>
        &dst) const;

    /**
     * Initialize a distributed vector. The local elements correspond to the
     * locally owned degrees of freedom and the ghost elements correspond to the
     * (additional) locally relevant dofs.
     */
    void
    initialize_dof_vector(
      dealii::LinearAlgebra::distributed::Vector<Number,
                                                 dealii::MemorySpace::CUDA>
        &vec) const;

    /**
     * Return the colored graph of locally owned active cells.
     */
    // const std::vector<std::vector<ActiveCellFilter>> &
    // get_colored_graph() const;

    /**
     * Return the partitioner that represents the locally owned data and the
     * ghost indices where access is needed to for the cell loop. The
     * partitioner is constructed from the locally owned dofs and ghost dofs
     * given by the respective fields. If you want to have specific information
     * about these objects, you can query them with the respective access
     * functions. If you just want to initialize a (parallel) vector, you should
     * usually prefer this data structure as the data exchange information can
     * be reused from one vector to another.
     */
    const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> &
    get_vector_partitioner() const;

    /**
     * Free all the memory allocated.
     */
    void
    free();

    /**
     * Return the DoFHandler.
     */
    const dealii::DoFHandler<dim> &
    get_dof_handler() const;

    /**
     * Return the number of dofs.
     */
    dealii::types::global_dof_index
    get_n_dofs() const;


    /**
     * Return an approximation of the memory consumption of this class in bytes.
     */
    std::size_t
    memory_consumption() const;

    MatrixType matrix_type;

  private:
    /**
     * Initializes the data structures.
     */
    template <bool is_MGLevel,
              typename CellIterator,
              typename IteratorFiltersType>
    void
    internal_reinit(const dealii::Mapping<dim>              &mapping,
                    const dealii::DoFHandler<dim>           &dof_handler,
                    const dealii::AffineConstraints<Number> &constraints,
                    const dealii::Quadrature<1>             &quad,
                    const IteratorFiltersType               &iterator_filter,
                    std::shared_ptr<const MPI_Comm>          comm,
                    const AdditionalData                     additional_data);

    /**
     * Helper function. Loop over all the cells and apply the functor on each
     * element in parallel. This function is used when MPI is not used.
     */
    template <typename Functor, typename VectorType>
    void
    serial_cell_loop(const Functor    &func,
                     const VectorType &src,
                     VectorType       &dst) const;

    /**
     * Helper function. Loop over all the cells and apply the functor on each
     * element in parallel. This function is used when MPI is used.
     */
    template <typename Functor>
    void
    distributed_cell_loop(
      const Functor &func,
      const dealii::LinearAlgebra::distributed::
        Vector<Number, dealii::MemorySpace::CUDA> &src,
      dealii::LinearAlgebra::distributed::Vector<Number,
                                                 dealii::MemorySpace::CUDA>
        &dst) const;

    /**
     * Unique ID associated with the object.
     */
    int my_id;

    /**
     * Stored the level of the mesh to be worked on.
     */
    unsigned int mg_level;

    /**
     * If true, use graph coloring. Otherwise, use atomic operations. Graph
     * coloring ensures bitwise reproducibility but is slower on Pascal and
     * newer architectures.
     */
    bool use_coloring;

    /**
     *  Overlap MPI communications with computation. This requires CUDA-aware
     *  MPI and use_coloring must be false.
     */
    bool overlap_communication_computation;

    /**
     * Total number of degrees of freedom.
     */
    dealii::types::global_dof_index n_dofs;

    /**
     * Degree of the finite element used.
     */
    unsigned int fe_degree;

    /**
     * Whether the entire finite element is primitive
     */
    bool is_primitive;

    /**
     * Number of components.
     */
    unsigned int n_components;

    /**
     * Number of degrees of freedom per cell.
     */
    unsigned int dofs_per_cell;

    /**
     * Number of degrees of freedom per face.
     */
    unsigned int dofs_per_face;

    /**
     * Number of constrained degrees of freedom.
     */
    unsigned int n_constrained_dofs;

    /**
     * Number of quadrature points per cell.
     */
    unsigned int q_points_per_cell;

    /**
     * Number of quadrature points per face.
     */
    unsigned int q_points_per_face;

    /**
     * Number of colors produced by the graph coloring algorithm.
     */
    unsigned int n_colors;

    /**
     * Number of cells in each color.
     */
    std::vector<unsigned int> n_cells;

    /**
     * Number of inner faces in each color.
     */
    std::vector<unsigned int> n_inner_faces;

    /**
     * Number of boundary faces in each color.
     */
    std::vector<unsigned int> n_boundary_faces;

    /**
     * Vector of pointers to the quadrature points associated to the cells of
     * each color.
     */
    std::vector<point_type *> q_points;

    /**
     * Map the position in the local vector to the position in the global
     * vector.
     */
    std::vector<dealii::types::global_dof_index *> local_to_global;

    /**
     * When using Multigrid Method, for one level coarse, map the position in
     * the local vector to the position in the global vector.
     */
    std::vector<dealii::types::global_dof_index *> l_to_g_coarse;

    /**
     * Map the inner face to cell id.
     */
    std::vector<dealii::types::global_dof_index *> inner_face2cell_id;

    /**
     * Map the boundary face to cell id.
     */
    std::vector<dealii::types::global_dof_index *> boundary_face2cell_id;

    /**
     * Vector of pointer to the cell Jacobian associated to the cells of each
     * color.
     */
    std::vector<Number *> jacobian;

    /**
     * Vector of pointer to the cell inverse Jacobian associated to the cells of
     * each color.
     */
    std::vector<Number *> inv_jacobian;

    /**
     * Vector of pointer to the cell Jacobian times the weights associated to
     * the cells of each color.
     */
    std::vector<Number *> JxW;

    /**
     * Vector of pointer to the cell quadrature weights associated to
     * the cells of each color.
     */
    std::vector<Number *> q_weights;

    /**
     * Vector of pointer to the determinant cell inverse Jacobian associated to
     * the cells of each color.
     */
    std::vector<Number *> inv_det;

    /**
     * Vector of pointer to the face Jacobian associated to the cells of each
     * color.
     */
    std::vector<Number *> face_jacobian;

    /**
     * Vector of pointer to the face inverse Jacobian associated to the cells of
     * each color.
     */
    std::vector<Number *> face_inv_jacobian;

    /**
     * Vector of pointer to the face Jacobian times the weights associated to
     * the cells of each color.
     */
    std::vector<Number *> face_JxW;

    /**
     * Vector of pointer to the face quadrature weights.
     */
    std::vector<Number *> face_q_weights;

    /**
     * Vector of pointer to the determinant face inverse Jacobian associated to
     * the cells of each color.
     */
    std::vector<Number *> face_inv_det;

    /**
     * Vector of pointer to the unit normal vector on a face associated to the
     * cells of each color.
     */
    std::vector<Number *> normal_vector;

    /**
     * Vector of pointer to the face direction.
     */
    std::vector<unsigned int *> face_number;

    /**
     * Vector of pointer to the subface number.
     */
    std::vector<int *> subface_number;

    /**
     * Vector of pointer to the face orientation.
     * Return whether the face with index face has its normal pointing in the
     * standard direction (true) or whether it is the opposite (false)
     */
    std::vector<int *> face_orientation;

    /**
     * Pointer to the constrained degrees of freedom.
     */
    dealii::types::global_dof_index *constrained_dofs;

    /**
     * Pointer to the hanging nodes constrained degrees of freedom.
     */
    dealii::types::global_dof_index *hanging_nodes_constraint;

    dealii::types::global_dof_index *hanging_nodes_constraint_coarse;

    /**
     * Pointer to the hanging nodes constraint indicator.
     */
    int *hanging_nodes_constraint_indicator;

    /**
     * Pointer to the hanging nodes constraint weights.
     */
    Number *hanging_nodes_constraint_weights;

    /**
     * Mask deciding where constraints are set on a given cell.
     */
    std::vector<dealii::internal::MatrixFreeFunctions::ConstraintKinds *>
      constraint_mask;

    /**
     * Range of cell constrains.
     */
    std::vector<dealii::types::global_dof_index *> constraint_range;
    std::vector<dealii::types::global_dof_index *> constraint_coarse_range;

    /**
     * Grid dimensions associated to the different colors. The grid dimensions
     * are used to launch the CUDA kernels.
     */
    std::vector<dim3> grid_dim;

    /**
     * Grid dimensions associated to the different colors. The grid dimensions
     * are used to launch the CUDA kernels.
     */
    std::vector<dim3> grid_dim_inner_face;

    /**
     * Grid dimensions associated to the different colors. The grid dimensions
     * are used to launch the CUDA kernels.
     */
    std::vector<dim3> grid_dim_boundary_face;

    /**
     * Block dimensions associated to the different colors. The block dimensions
     * are used to launch the CUDA kernels.
     */
    std::vector<dim3> block_dim;

    /**
     * Block dimensions associated to the different colors. The block dimensions
     * are used to launch the CUDA kernels.
     */
    std::vector<dim3> block_dim_inner_face;

    /**
     * Block dimensions associated to the different colors. The block dimensions
     * are used to launch the CUDA kernels.
     */
    std::vector<dim3> block_dim_boundary_face;

    /**
     * Shared pointer to a Partitioner for distributed Vectors used in
     * cell_loop. When MPI is not used the pointer is null.
     */
    std::shared_ptr<const dealii::Utilities::MPI::Partitioner> partitioner;

    /**
     * Cells per block (determined by the function cells_per_block_shmem() ).
     */
    unsigned int cells_per_block;

    /**
     * Boundary faces per block (determined by the function todo() ).
     */
    unsigned int boundary_faces_per_block;

    /**
     * Inner faces per block (determined by the function todo() ).
     */
    unsigned int inner_faces_per_block;

    /**
     * Grid dimensions used to launch the CUDA kernels
     * in *_constrained_values-operations.
     */
    dim3 constraint_grid_dim;

    /**
     * Block dimensions used to launch the CUDA kernels
     * in *_constrained_values-operations.
     */
    dim3 constraint_block_dim;

    /**
     * Length of the padding (closest power of two larger than or equal to
     * the number of thread).
     */
    unsigned int padding_length;

    /**
     * Length of the face padding (closest power of two larger than or equal to
     * the number of thread).
     */
    unsigned int face_padding_length;

    /**
     * Row start of each color.
     */
    std::vector<unsigned int> row_start;

    /**
     * Pointer to the DoFHandler associated with the object.
     */
    const dealii::DoFHandler<dim> *dof_handler;

    /**
     * Colored graphed of locally owned active cells.
     */
    // std::vector<std::vector<ActiveCellFilter>> graph;

    /**
     * Colored graphed of locally owned level cells.
     */
    // std::vector<std::vector<LevelCellFilter>> level_graph;

    friend class ReinitHelper<dim, Number>;
  };


  /**
   * Structure to pass the shared memory into a general user function.
   */
  template <int dim, typename Number>
  struct SharedData
  {
    /**
     * Constructor.
     */
    __device__
    SharedData(Number *vd, Number *gq[dim])
      : values(vd)
    {
      for (unsigned int d = 0; d < dim; ++d)
        gradients[d] = gq[d];
    }

    /**
     * Shared memory for dof and quad values.
     */
    Number *values;

    /**
     * Shared memory for computed gradients in reference coordinate system.
     * The gradient in each direction is saved in a struct-of-array
     * format, i.e. first, all gradients in the x-direction come...
     */
    Number *gradients[dim];
  };

  // This function determines the number of cells per block, possibly at compile
  // time (by virtue of being 'constexpr')
  // TODO this function should be rewritten using meta-programming
  __host__ __device__ constexpr unsigned int
  cells_per_block_shmem(int dim, int fe_degree)
  {
    return 1;

    constexpr int warp_size = 32;

    /* clang-format off */
    // We are limiting the number of threads according to the
    // following formulas:
    //  - in 2D: `threads = cells * (k+1)^d <= 4*warp_size`
    //  - in 3D: `threads = cells * (k+1)^d <= 2*warp_size`
    return dim==2 ? (fe_degree==1 ? warp_size :    // 128
                     fe_degree==2 ? warp_size/4 :  //  72
                     fe_degree==3 ? warp_size/8 :  //  64
                     fe_degree==4 ? warp_size/8 :  // 100
                     1) :
           dim==3 ? (fe_degree==1 ? warp_size/4 :  //  64
                     fe_degree==2 ? warp_size/16 : //  54
                     1) : 1;
    /* clang-format on */
  }


  /*----------------------- Helper functions ---------------------------------*/
  /**
   * Compute the quadrature point index in the local cell of a given thread.
   *
   * @relates MatrixFree
   */
  template <int dim>
  __device__ inline unsigned int
  q_point_id_in_cell(const unsigned int n_q_points_1d)
  {
    return (
      dim == 1 ? threadIdx.x % n_q_points_1d :
      dim == 2 ? threadIdx.x % n_q_points_1d + n_q_points_1d * threadIdx.y :
                 threadIdx.x % n_q_points_1d +
                   n_q_points_1d * (threadIdx.y + n_q_points_1d * threadIdx.z));
  }

  /**
   * Return the quadrature point index local of a given thread. The index is
   * only unique for a given MPI process.
   *
   * @relates MatrixFree
   */
  template <int dim, typename Number>
  __device__ inline unsigned int
  local_q_point_id(const unsigned int                            cell,
                   const typename MatrixFree<dim, Number>::Data *data,
                   const unsigned int                            n_q_points_1d,
                   const unsigned int                            n_q_points)
  {
    return (data->row_start / data->padding_length + cell) * n_q_points +
           q_point_id_in_cell<dim>(n_q_points_1d);
  }

  /**
   * Return the quadrature point associated with a given thread.
   *
   * @relates MatrixFree
   */
  template <int dim, typename Number>
  __device__ inline typename MatrixFree<dim, Number>::point_type &
  get_quadrature_point(const unsigned int                            cell,
                       const typename MatrixFree<dim, Number>::Data *data,
                       const unsigned int n_q_points_1d)
  {
    return *(data->q_points + data->padding_length * cell +
             q_point_id_in_cell<dim>(n_q_points_1d));
  }

  /*----------------------- Inline functions ---------------------------------*/
  // template <int dim, typename Number>
  // inline const std::vector<std::vector<dealii::FilteredIterator<
  //   typename dealii::DoFHandler<dim>::active_cell_iterator>>> &
  // MatrixFree<dim, Number>::get_colored_graph() const
  // {
  //   return graph;
  // }



  template <int dim, typename Number>
  inline const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> &
  MatrixFree<dim, Number>::get_vector_partitioner() const
  {
    return partitioner;
  }



  template <int dim, typename Number>
  inline const dealii::DoFHandler<dim> &
  MatrixFree<dim, Number>::get_dof_handler() const
  {
    Assert(dof_handler != nullptr, dealii::ExcNotInitialized());

    return *dof_handler;
  }


  template <int dim, typename Number>
  inline dealii::types::global_dof_index
  MatrixFree<dim, Number>::get_n_dofs() const
  {
    return n_dofs;
  }

} // namespace PSMF

#include "cuda_matrix_free.template.cuh"

#endif // CUDA_MATRIX_FREE_CUH