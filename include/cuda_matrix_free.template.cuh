/**
 * @file cuda_matrix_free.template.cuh
 * @brief Matrix free class.
 *
 * This class collects all the data that is stored for the matrix free
 * implementation.
 *
 * @author Cu Cui
 * @date 2024-01-19
 * @version 0.1
 *
 * @remark
 * @note
 * @warning
 */


// #ifndef CUDA_MATRIX_FREE_TEMPLATE_CUH
// #define CUDA_MATRIX_FREE_TEMPLATE_CUH

#include <deal.II/base/config.h>

#include <deal.II/base/graph_coloring.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/matrix_free/cuda_hanging_nodes_internal.h>
#include <deal.II/matrix_free/shape_info.h>

#include <cuda_runtime_api.h>

#include <cmath>
#include <functional>
#include <type_traits>

// #include "cuda_matrix_free.cuh"
#include "renumber.h"

namespace PSMF
{

  /**
   * Define the size of a block when launching a CUDA kernel. This number can be
   * changed depending on the architecture the code is running on.
   */
  constexpr int block_size = 512;

  /**
   * Define the maximum number of valid MatrixFree object.
   * Changing this number will affect the amount of constant memory being used.
   */
  constexpr unsigned int mf_n_concurrent_objects = 2;

  /**
   * Define the largest finite element degree that can be solved using
   * MatrixFree. Changing this number will affect the amount of
   * constant memory being used.
   */
  constexpr unsigned int mf_max_elem_degree = 15;

  constexpr unsigned int data_array_size =
    (mf_max_elem_degree + 1) * (mf_max_elem_degree + 1);

  // Default initialized to false
  std::array<std::atomic_bool, mf_n_concurrent_objects> used_objects;

  template <typename NumberType>
  using DataArray = NumberType[data_array_size];

  // These variables are stored in the device constant memory.
  // Shape values
  __constant__ double cell_shape_values_d[mf_n_concurrent_objects]
                                         [data_array_size];
  __constant__ float cell_shape_values_f[mf_n_concurrent_objects]
                                        [data_array_size];
  // Collects all data of 1D shape values evaluated at the point 0 and 1 (the
  // vertices) in one data structure. Sorting is first the values, then
  // gradients, then second derivatives.
  __constant__ double face_shape_values_d[mf_n_concurrent_objects]
                                         [data_array_size];
  __constant__ float face_shape_values_f[mf_n_concurrent_objects]
                                        [data_array_size];
  // Shape gradients
  __constant__ double cell_shape_gradients_d[mf_n_concurrent_objects]
                                            [data_array_size];
  __constant__ float cell_shape_gradients_f[mf_n_concurrent_objects]
                                           [data_array_size];
  __constant__ double face_shape_gradients_d[mf_n_concurrent_objects]
                                            [data_array_size];
  __constant__ float face_shape_gradients_f[mf_n_concurrent_objects]
                                           [data_array_size];
  // for collocation methods
  __constant__ double cell_co_shape_gradients_d[mf_n_concurrent_objects]
                                               [data_array_size];
  __constant__ float cell_co_shape_gradients_f[mf_n_concurrent_objects]
                                              [data_array_size];
  __constant__ double face_co_shape_gradients_d[mf_n_concurrent_objects]
                                               [data_array_size];
  __constant__ float face_co_shape_gradients_f[mf_n_concurrent_objects]
                                              [data_array_size];


  template <typename Number>
  __host__ __device__ inline DataArray<Number>          &
  get_cell_shape_values(unsigned int i);

  template <>
  __host__ __device__ inline DataArray<double>          &
  get_cell_shape_values<double>(unsigned int i)
  {
    return cell_shape_values_d[i];
  }

  template <>
  __host__ __device__ inline DataArray<float>          &
  get_cell_shape_values<float>(unsigned int i)
  {
    return cell_shape_values_f[i];
  }

  template <typename Number>
  __host__ __device__ inline DataArray<Number>          &
  get_cell_shape_gradients(unsigned int i);

  template <>
  __host__ __device__ inline DataArray<double>          &
  get_cell_shape_gradients<double>(unsigned int i)
  {
    return cell_shape_gradients_d[i];
  }

  template <>
  __host__ __device__ inline DataArray<float>          &
  get_cell_shape_gradients<float>(unsigned int i)
  {
    return cell_shape_gradients_f[i];
  }

  // for collocation methods
  template <typename Number>
  __host__ __device__ inline DataArray<Number>          &
  get_cell_co_shape_gradients(unsigned int i);

  template <>
  __host__ __device__ inline DataArray<double>          &
  get_cell_co_shape_gradients<double>(unsigned int i)
  {
    return cell_co_shape_gradients_d[i];
  }

  template <>
  __host__ __device__ inline DataArray<float>          &
  get_cell_co_shape_gradients<float>(unsigned int i)
  {
    return cell_co_shape_gradients_f[i];
  }



  template <typename Number>
  __host__ __device__ inline DataArray<Number>          &
  get_face_shape_values(unsigned int i);

  template <>
  __host__ __device__ inline DataArray<double>          &
  get_face_shape_values<double>(unsigned int i)
  {
    return face_shape_values_d[i];
  }

  template <>
  __host__ __device__ inline DataArray<float>          &
  get_face_shape_values<float>(unsigned int i)
  {
    return face_shape_values_f[i];
  }

  template <typename Number>
  __host__ __device__ inline DataArray<Number>          &
  get_face_shape_gradients(unsigned int i);

  template <>
  __host__ __device__ inline DataArray<double>          &
  get_face_shape_gradients<double>(unsigned int i)
  {
    return face_shape_gradients_d[i];
  }

  template <>
  __host__ __device__ inline DataArray<float>          &
  get_face_shape_gradients<float>(unsigned int i)
  {
    return face_shape_gradients_f[i];
  }


  // for collocation methods
  template <typename Number>
  __host__ __device__ inline DataArray<Number>          &
  get_face_co_shape_gradients(unsigned int i);

  template <>
  __host__ __device__ inline DataArray<double>          &
  get_face_co_shape_gradients<double>(unsigned int i)
  {
    return face_co_shape_gradients_d[i];
  }

  template <>
  __host__ __device__ inline DataArray<float>          &
  get_face_co_shape_gradients<float>(unsigned int i)
  {
    return face_co_shape_gradients_f[i];
  }



  /**
   * Transpose a N x M matrix stored in a one-dimensional array to a M x N
   * matrix stored in a one-dimensional array.
   */
  template <typename Number>
  void
  transpose(const unsigned int N,
            const unsigned     M,
            const Number      *src,
            Number            *dst)
  {
    // src is N X M
    // dst is M X N
    for (unsigned int i = 0; i < N; ++i)
      for (unsigned int j = 0; j < M; ++j)
        dst[j * N + i] = src[i * M + j];
  }


  /**
   * Same as above but the source and the destination are the same vector.
   */
  template <typename Number>
  void
  transpose_in_place(std::vector<Number> &array_host,
                     const unsigned int   n,
                     const unsigned int   m)
  {
    // convert to structure-of-array
    std::vector<Number> old(array_host.size());
    old.swap(array_host);

    transpose(n, m, old.data(), array_host.data());
  }


  /**
   * Allocate an array to the device and copy @p array_host to the device.
   */
  template <typename Number1>
  void
  alloc_and_copy(Number1 **array_device,
                 const dealii::ArrayView<const Number1,
                                         dealii::MemorySpace::Host> array_host,
                 const unsigned int                                 n)
  {
    cudaError_t error_code = cudaMalloc(array_device, n * sizeof(Number1));
    AssertCuda(error_code);
    AssertDimension(array_host.size(), n);

    error_code = cudaMemcpy(*array_device,
                            array_host.data(),
                            n * sizeof(Number1),
                            cudaMemcpyHostToDevice);
    AssertCuda(error_code);
  }



  /**
   * Helper class to (re)initialize MatrixFree object.
   */
  template <int dim, typename Number>
  class ReinitHelper
  {
  public:
    using ActiveCellFilter = dealii::FilteredIterator<
      typename dealii::DoFHandler<dim>::active_cell_iterator>;
    using LevelCellFilter = dealii::FilteredIterator<
      typename dealii::DoFHandler<dim>::level_cell_iterator>;

    ReinitHelper(MatrixFree<dim, Number>               *data,
                 const dealii::Mapping<dim>            &mapping,
                 const dealii::FiniteElement<dim, dim> &fe,
                 const dealii::Quadrature<1>           &quad,
                 const dealii::internal::MatrixFreeFunctions::ShapeInfo<Number>
                                                         &shape_info,
                 const dealii::DoFHandler<dim>           &dof_handler,
                 const dealii::AffineConstraints<Number> &constraints,
                 const dealii::UpdateFlags               &update_flags,
                 const dealii::UpdateFlags &update_flags_boundary_faces =
                   dealii::update_default,
                 const dealii::UpdateFlags &update_flags_inner_faces =
                   dealii::update_default);

    void
    setup_color_arrays(const unsigned int n_colors);

    void
    setup_cell_arrays(const unsigned int color);

    void
    setup_face_arrays(const unsigned int color);

    template <typename CellFilter>
    void
    get_cell_data(
      const CellFilter &cell,
      unsigned int     &cell_id,
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
                        &partitioner,
      const unsigned int color);

    template <typename CellFilter>
    void
    get_face_data(CellFilter        &cell,
                  unsigned int      &inner_face_id,
                  unsigned int      &boundary_face_id,
                  const unsigned int color);

    void
    alloc_and_copy_arrays(const unsigned int color);

    void
    alloc_and_copy_face_arrays(const unsigned int color);

    unsigned int n_inner_faces;
    unsigned int n_boundary_faces;

  private:
    MatrixFree<dim, Number> *data;
    // Host data
    std::vector<dealii::types::global_dof_index> local_to_global_host;
    std::vector<dealii::types::global_dof_index> l_to_g_coarse_host;
    std::vector<dealii::types::global_dof_index> inner_face2cell_id_host;
    std::vector<dealii::types::global_dof_index> boundary_face2cell_id_host;
    std::vector<dealii::Point<dim, Number>>      q_points_host;
    std::vector<Number>                          JxW_host;
    std::vector<Number>                          q_weights_host;
    std::vector<Number>                          inv_det_host;
    std::vector<Number>                          jacobian_host;
    std::vector<Number>                          inv_jacobian_host;
    std::vector<Number>                          face_JxW_host;
    std::vector<Number>                          face_q_weights_host;
    std::vector<Number>                          face_inv_det_host;
    std::vector<Number>                          face_jacobian_host;
    std::vector<Number>                          face_inv_jacobian_host;
    std::vector<Number>                          normal_vector_host;
    std::vector<dealii::types::global_dof_index> face_number_host;
    std::vector<int>                             subface_number_host;
    std::vector<int>                             face_orientation_host;
    std::vector<dealii::internal::MatrixFreeFunctions::ConstraintKinds>
                                                 constraint_mask_host;
    std::vector<dealii::types::global_dof_index> constraint_range_host;
    std::vector<dealii::types::global_dof_index> constraint_coarse_range_host;
    // Local buffer
    std::vector<dealii::types::global_dof_index> local_dof_indices;
    std::vector<dealii::types::global_dof_index> local_dof_indices_coarse;
    dealii::FEValues<dim>                        fe_values;
    dealii::FEFaceValues<dim>                    fe_face_values;
    dealii::FESubfaceValues<dim>                 fe_subface_values;
    // Convert the default dof numbering to a lexicographic one
    std::vector<unsigned int>                    lexicographic_inv;
    std::vector<dealii::types::global_dof_index> lexicographic_dof_indices;
    std::vector<unsigned int>                    constrained_indices;
    std::vector<unsigned int>                    constrained_indices_coarse;
    const unsigned int                           fe_degree;
    const unsigned int                           n_components;
    const unsigned int                           dofs_per_cell;
    const unsigned int                           dofs_per_face;
    const unsigned int                           q_points_per_cell;
    const unsigned int                           q_points_per_face;
    const dealii::UpdateFlags                   &update_flags;
    const dealii::UpdateFlags                   &update_flags_boundary_faces;
    const dealii::UpdateFlags                   &update_flags_inner_faces;
    const unsigned int                           padding_length;
    const unsigned int                           face_padding_length;
    const bool                                   element_is_continuous;
    const bool                                   is_primitive;
    const unsigned int                           mg_level;
    const MatrixType                             matrix_type;
    const dealii::AffineConstraints<Number>     &constraints;
    dealii::internal::MatrixFreeFunctions::HangingNodes<dim> hanging_nodes;
    std::vector<std::map<std::pair<int, int>, int>>          cell2cell_id;
    std::vector<std::map<std::pair<int, int>, int>> cell2cell_id_coarse;
  };

  template <int dim, typename Number>
  ReinitHelper<dim, Number>::ReinitHelper(
    MatrixFree<dim, Number>                                        *data,
    const dealii::Mapping<dim>                                     &mapping,
    const dealii::FiniteElement<dim>                               &fe,
    const dealii::Quadrature<1>                                    &quad,
    const dealii::internal::MatrixFreeFunctions::ShapeInfo<Number> &shape_info,
    const dealii::DoFHandler<dim>                                  &dof_handler,
    const dealii::AffineConstraints<Number>                        &constraints,
    const dealii::UpdateFlags &update_flags,
    const dealii::UpdateFlags &update_flags_inner_faces,
    const dealii::UpdateFlags &update_flags_boundary_faces)
    : data(data)
    , fe_degree(data->fe_degree)
    , n_components(data->n_components)
    , dofs_per_cell(data->dofs_per_cell)
    , dofs_per_face(data->dofs_per_face)
    , q_points_per_cell(data->q_points_per_cell)
    , q_points_per_face(data->q_points_per_face)
    , fe_values(mapping,
                fe,
                dealii::Quadrature<dim>(quad),
                dealii::update_inverse_jacobians | dealii::update_jacobians |
                  dealii::update_quadrature_points | dealii::update_values |
                  dealii::update_gradients | dealii::update_JxW_values)
    , fe_face_values(mapping,
                     fe,
                     dealii::Quadrature<dim - 1>(quad),
                     dealii::update_inverse_jacobians |
                       dealii::update_jacobians |
                       dealii::update_quadrature_points |
                       dealii::update_normal_vectors | dealii::update_values |
                       dealii::update_gradients | dealii::update_JxW_values)
    , fe_subface_values(mapping,
                        fe,
                        dealii::Quadrature<dim - 1>(quad),
                        dealii::update_inverse_jacobians |
                          dealii::update_jacobians |
                          dealii::update_quadrature_points |
                          dealii::update_normal_vectors |
                          dealii::update_values | dealii::update_gradients |
                          dealii::update_JxW_values)
    , update_flags(update_flags)
    , update_flags_inner_faces(update_flags_inner_faces)
    , update_flags_boundary_faces(update_flags_boundary_faces)
    , padding_length(data->get_padding_length())
    , face_padding_length(data->get_face_padding_length())
    , element_is_continuous(fe.n_dofs_per_vertex() > 0)
    , is_primitive(data->is_primitive)
    , mg_level(data->mg_level)
    , matrix_type(data->matrix_type)
    , constraints(constraints)
    , n_inner_faces(0)
    , n_boundary_faces(0)
    , hanging_nodes(dof_handler.get_triangulation())
  {
    // cudaError_t error_code = cudaMemcpyToSymbol(
    //   dealii::CUDAWrappers::internal::constraint_weights,
    //   shape_info.data.front().subface_interpolation_matrices[0].data(),
    //   sizeof(double) *
    //     shape_info.data.front().subface_interpolation_matrices[0].size());
    // AssertCuda(error_code);

    auto lines = constraints.get_lines();
    for (auto &l : lines)
      if (l.entries.size() > 0)
        for (auto e : l.entries)
          if (e.second < -100)
            constrained_indices_coarse.push_back(e.first);
          else
            constrained_indices.push_back(l.index);
      else
        constrained_indices.push_back(l.index);

    if (is_primitive)
      {
        lexicographic_inv = shape_info.lexicographic_numbering;
      }
    else
      {
        lexicographic_inv = shape_info.lexicographic_numbering;

        // DoFMapping<dim> dm(fe_degree - 1);
        // lexicographic_inv = dm.get_lexicographic_numbering();
      }

    local_dof_indices.resize(data->dofs_per_cell);
    local_dof_indices_coarse.resize(data->dofs_per_cell);
    lexicographic_dof_indices.resize(data->dofs_per_cell);

    // std::cout << "constrained_indices_coarse\n";
    // for (auto c : constrained_indices_coarse)
    //   std::cout << c << " ";
    // std::cout << std::endl;
  }

  template <int dim, typename Number>
  void
  ReinitHelper<dim, Number>::setup_color_arrays(const unsigned int n_colors)
  {
    cell2cell_id.resize(n_colors);
    cell2cell_id_coarse.resize(n_colors);

    // We need at least three colors when we are using CUDA-aware MPI and
    // overlapping the communication
    data->n_cells.resize(std::max(n_colors, 3U), 0);
    data->n_inner_faces.resize(std::max(n_colors, 3U), 0);
    data->n_boundary_faces.resize(std::max(n_colors, 3U), 0);
    data->grid_dim.resize(n_colors);
    data->grid_dim_inner_face.resize(n_colors);
    data->grid_dim_boundary_face.resize(n_colors);
    data->block_dim.resize(n_colors);
    data->block_dim_inner_face.resize(n_colors);
    data->block_dim_boundary_face.resize(n_colors);
    data->local_to_global.resize(n_colors);
    data->l_to_g_coarse.resize(n_colors);
    data->inner_face2cell_id.resize(n_colors);
    data->boundary_face2cell_id.resize(n_colors);
    data->face_number.resize(n_colors);
    data->subface_number.resize(n_colors);
    data->face_orientation.resize(n_colors);
    data->constraint_mask.resize(n_colors);
    data->constraint_range.resize(n_colors);
    data->constraint_coarse_range.resize(n_colors);

    data->row_start.resize(n_colors);

    if (update_flags & dealii::update_quadrature_points)
      data->q_points.resize(n_colors);

    if (update_flags & dealii::update_JxW_values)
      data->JxW.resize(n_colors);

    if (update_flags & dealii::update_jacobians)
      {
        data->jacobian.resize(n_colors);
        data->inv_det.resize(n_colors);
        data->q_weights.resize(n_colors);
      }

    if (update_flags & dealii::update_gradients)
      data->inv_jacobian.resize(n_colors);

    if (update_flags_inner_faces & dealii::update_JxW_values)
      data->face_JxW.resize(n_colors);

    if (update_flags_inner_faces & dealii::update_gradients)
      data->face_inv_jacobian.resize(n_colors);

    if (update_flags_inner_faces & dealii::update_jacobians)
      {
        data->face_jacobian.resize(n_colors);
        data->face_inv_det.resize(n_colors);
        data->face_q_weights.resize(n_colors);
      }

    if (update_flags_inner_faces & dealii::update_normal_vectors)
      data->normal_vector.resize(n_colors);
  }

  template <int dim, typename Number>
  void
  ReinitHelper<dim, Number>::setup_cell_arrays(const unsigned int color)
  {
    const unsigned int n_cells         = data->n_cells[color];
    const unsigned int cells_per_block = data->cells_per_block;

    // Setup kernel parameters
    double apply_n_blocks = std::ceil(static_cast<double>(n_cells) /
                                      static_cast<double>(cells_per_block));
    data->grid_dim[color] = dim3(apply_n_blocks);

    // TODO this should be a templated parameter.
    const unsigned int n_dofs_1d = fe_degree + 1;

    if (dim == 2)
      {
        data->block_dim[color] = dim3(n_dofs_1d * cells_per_block, n_dofs_1d);
      }
    else if (dim == 3)
      {
        data->block_dim[color] =
          dim3(n_dofs_1d * cells_per_block, n_dofs_1d, n_dofs_1d);
      }
    else
      AssertThrow(false, dealii::ExcMessage("Invalid dimension."));


    if (matrix_type == MatrixType::edge_up_matrix ||
        matrix_type == MatrixType::edge_down_matrix)
      {
        local_to_global_host.resize(n_cells * padding_length * n_components *
                                    (1 << dim - 1));
        l_to_g_coarse_host.resize(n_cells * padding_length * n_components *
                                  (1 << dim - 1));
      }
    else if (matrix_type == MatrixType::interface_matrix)
      {
        local_to_global_host.resize(n_cells * padding_length * n_components *
                                    (1 << dim));
        l_to_g_coarse_host.resize(n_cells * padding_length * n_components *
                                  (1 << dim));
      }
    else
      {
        local_to_global_host.resize(n_cells * padding_length * n_components);
        l_to_g_coarse_host.resize(n_cells * padding_length * n_components);
      }

    if (update_flags & dealii::update_quadrature_points)
      q_points_host.resize(n_cells * padding_length);

    if (update_flags & dealii::update_JxW_values)
      JxW_host.resize(n_cells * padding_length);

    if (update_flags & dealii::update_jacobians)
      {
        jacobian_host.resize(n_cells * padding_length * dim * dim);
        inv_det_host.resize(n_cells * padding_length);
        q_weights_host.resize(padding_length);
      }

    if (update_flags & dealii::update_gradients)
      inv_jacobian_host.resize(n_cells * padding_length * dim * dim);

    constraint_mask_host.resize(n_cells * n_components);
    constraint_range_host.resize(n_cells * 2 * (1 << dim - 1));
    constraint_coarse_range_host.resize(n_cells * 2 * (1 << dim - 1));
  }


  template <int dim, typename Number>
  void
  ReinitHelper<dim, Number>::setup_face_arrays(const unsigned int color)
  {
    const unsigned int n_inner_faces         = data->n_inner_faces[color];
    const unsigned int inner_faces_per_block = data->inner_faces_per_block;

    const unsigned int n_boundary_faces = data->n_boundary_faces[color];
    const unsigned int boundary_faces_per_block =
      data->boundary_faces_per_block;

    const unsigned int n_faces = n_inner_faces * 2 + n_boundary_faces;

    // Setup kernel parameters
    double apply_n_blocks =
      std::ceil(static_cast<double>(n_inner_faces) /
                static_cast<double>(inner_faces_per_block));
    data->grid_dim_inner_face[color] = dim3(apply_n_blocks);

    apply_n_blocks = std::ceil(static_cast<double>(n_boundary_faces) /
                               static_cast<double>(boundary_faces_per_block));
    data->grid_dim_boundary_face[color] = dim3(apply_n_blocks);

    // TODO this should be a templated parameter.
    const unsigned int n_dofs_1d = fe_degree + 1;

    if (dim == 2)
      {
        data->block_dim_inner_face[color] =
          dim3(n_dofs_1d * inner_faces_per_block, n_dofs_1d);
        data->block_dim_boundary_face[color] =
          dim3(n_dofs_1d * boundary_faces_per_block, n_dofs_1d);
      }
    else if (dim == 3)
      {
        data->block_dim_inner_face[color] =
          dim3(n_dofs_1d * inner_faces_per_block, n_dofs_1d, n_dofs_1d);
        data->block_dim_boundary_face[color] =
          dim3(n_dofs_1d * boundary_faces_per_block, n_dofs_1d, n_dofs_1d);
      }
    else
      AssertThrow(false, dealii::ExcMessage("Invalid dimension."));

    inner_face2cell_id_host.resize(n_inner_faces * 2);
    boundary_face2cell_id_host.resize(n_boundary_faces);
    face_number_host.resize(n_faces);
    subface_number_host.resize(n_faces);
    face_orientation_host.resize(n_faces);

    if (update_flags_inner_faces & dealii::update_JxW_values)
      face_JxW_host.resize(n_faces * face_padding_length);

    if (update_flags_inner_faces & dealii::update_gradients)
      face_inv_jacobian_host.resize(n_faces * face_padding_length * dim * dim);

    if (update_flags_inner_faces & dealii::update_jacobians)
      {
        face_jacobian_host.resize(n_faces * face_padding_length * dim * dim);
        face_inv_det_host.resize(n_faces * face_padding_length);
        face_q_weights_host.resize(face_padding_length);
      }

    if (update_flags_inner_faces & dealii::update_normal_vectors)
      normal_vector_host.resize(n_faces * face_padding_length * dim * 1);
  }

  template <int dim, typename Number>
  template <typename CellFilter>
  void
  ReinitHelper<dim, Number>::get_cell_data(
    const CellFilter &cell,
    unsigned int     &cell_id,
    const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
                      &partitioner,
    const unsigned int color)
  {
    auto fill_index_data = [&](auto &c, auto obj_id) {
      c->get_active_or_mg_dof_indices(local_dof_indices);

      // When using MPI, we need to transform the local_dof_indices, which
      // contains global dof indices, to get local (to the current MPI
      // process) dof indices.
      if (partitioner)
        for (auto &index : local_dof_indices)
          index = partitioner->global_to_local(index);


      constraint_range_host[2 * obj_id]     = 0;
      constraint_range_host[2 * obj_id + 1] = 0;

      unsigned int begin_pos = constrained_indices.size() + 1;
      unsigned int n_cons    = 0;

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          lexicographic_dof_indices[i] =
            local_dof_indices[lexicographic_inv[i]];

          if (constraints.is_constrained(lexicographic_dof_indices[i]))
            {
              auto entries_size =
                constraints
                  .get_constraint_entries(lexicographic_dof_indices[i])
                  ->size();

              auto         it   = std::find(constrained_indices.begin(),
                                  constrained_indices.end(),
                                  lexicographic_dof_indices[i]);
              unsigned int diff = it - constrained_indices.begin();
              begin_pos         = std::min(begin_pos, diff);
              n_cons += (entries_size > 0 ? entries_size : 1);

              constraint_range_host[2 * obj_id] = begin_pos;
            }
        }
      constraint_range_host[2 * obj_id + 1] =
        constraint_range_host[2 * obj_id] + n_cons;

      // if (obj_id >= 5)
      //   {
      //     std::cout << obj_id << ": " << constraint_range_host[2 * obj_id]
      //               << " " << constraint_range_host[2 * obj_id + 1]
      //               << std::endl;
      //   }

      // std::cout << obj_id << ": " << constraint_range_host[2 * obj_id] << " "
      //           << constraint_range_host[2 * obj_id + 1] << std::endl;

      // const dealii::ArrayView<
      //   dealii::internal::MatrixFreeFunctions::ConstraintKinds>
      //   cell_id_view(&constraint_mask_host[obj_id * n_components],
      //                n_components);

      // hanging_nodes.setup_constraints(c,
      //                                 partitioner,
      //                                 {lexicographic_inv},
      //                                 lexicographic_dof_indices,
      //                                 cell_id_view);

      memcpy(&local_to_global_host[obj_id * padding_length * n_components],
             lexicographic_dof_indices.data(),
             dofs_per_cell * sizeof(dealii::types::global_dof_index));
    };

    auto fill_cell_data = [&](auto &fe_value, auto obj_id) {
      // Quadrature points
      if (update_flags & dealii::update_quadrature_points)
        {
          const std::vector<dealii::Point<dim>> &q_points =
            fe_value.get_quadrature_points();
          std::copy(q_points.begin(),
                    q_points.end(),
                    &q_points_host[obj_id * padding_length]);
        }

      if (update_flags & dealii::update_JxW_values)
        {
          std::vector<double> JxW_values_double = fe_value.get_JxW_values();
          const unsigned int  offset            = obj_id * padding_length;
          for (unsigned int i = 0; i < q_points_per_cell; ++i)
            JxW_host[i + offset] = static_cast<Number>(JxW_values_double[i]);
        }

      if (update_flags & dealii::update_jacobians)
        {
          const std::vector<dealii::DerivativeForm<1, dim, dim>> &jacobians =
            fe_value.get_jacobians();
          std::copy(&jacobians[0][0][0],
                    &jacobians[0][0][0] +
                      q_points_per_cell *
                        sizeof(dealii::DerivativeForm<1, dim, dim>) /
                        sizeof(double),
                    &jacobian_host[obj_id * padding_length * dim * dim]);

          // determinant of inverse jacobian
          const unsigned int offset = obj_id * padding_length;
          const std::vector<dealii::DerivativeForm<1, dim, dim>>
            &inv_jacobians = fe_value.get_inverse_jacobians();
          for (unsigned int i = 0; i < q_points_per_cell; ++i)
            inv_det_host[i + offset] =
              static_cast<Number>(inv_jacobians[i].determinant());

          // quadrature weights
          if (obj_id == 0)
            {
              std::vector<double> quadrature_weights =
                fe_value.get_quadrature().get_weights();
              for (unsigned int i = 0; i < q_points_per_cell; ++i)
                q_weights_host[i] = static_cast<Number>(quadrature_weights[i]);
            }
        }

      if (update_flags & dealii::update_gradients)
        {
          const std::vector<dealii::DerivativeForm<1, dim, dim>>
            &inv_jacobians = fe_value.get_inverse_jacobians();
          std::copy(&inv_jacobians[0][0][0],
                    &inv_jacobians[0][0][0] +
                      q_points_per_cell *
                        sizeof(dealii::DerivativeForm<1, dim, dim>) /
                        sizeof(double),
                    &inv_jacobian_host[obj_id * padding_length * dim * dim]);
        }
    };


    if (matrix_type == MatrixType::active_matrix ||
        matrix_type == MatrixType::level_matrix ||
        matrix_type == MatrixType::interface_matrix)
      {
        auto cell_info = std::make_pair<int, int>(cell->level(), cell->index());
        cell2cell_id[color][cell_info] = cell_id;

        fill_index_data(cell, cell_id);

        fe_values.reinit(cell);
        fill_cell_data(fe_values, cell_id);

        cell_id++;
      }

    for (const unsigned int face_no : cell->face_indices())
      {
        if (cell->at_boundary(face_no))
          {
            n_boundary_faces++;
          }
        else
          {
            auto neighbor = cell->neighbor_or_periodic_neighbor(face_no);

            if (cell->neighbor_is_coarser(face_no))
              {
                if (matrix_type == MatrixType::edge_up_matrix ||
                    matrix_type == MatrixType::edge_down_matrix)
                  {
                    auto cell_info =
                      std::make_pair<int, int>(cell->level(), cell->index());
                    cell2cell_id[color][cell_info] = cell_id;

                    fill_index_data(cell, cell_id);

                    auto cell_info_coarse =
                      std::make_pair<int, int>(neighbor->level(),
                                               neighbor->index());
                    cell2cell_id_coarse[color][cell_info_coarse] = cell_id;

                    neighbor->get_active_or_mg_dof_indices(
                      local_dof_indices_coarse);

                    constraint_coarse_range_host[2 * cell_id]     = 0;
                    constraint_coarse_range_host[2 * cell_id + 1] = 0;

                    unsigned int begin_pos =
                      constrained_indices_coarse.size() + 1;
                    unsigned int n_cons = 0;

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        lexicographic_dof_indices[i] =
                          local_dof_indices_coarse[lexicographic_inv[i]];

                        auto it = std::find(constrained_indices_coarse.begin(),
                                            constrained_indices_coarse.end(),
                                            lexicographic_dof_indices[i]);

                        if (it == constrained_indices_coarse.end())
                          continue;

                        unsigned int diff =
                          it - constrained_indices_coarse.begin();
                        begin_pos = std::min(begin_pos, diff);
                        n_cons += 1;

                        constraint_coarse_range_host[2 * cell_id] = begin_pos;
                      }
                    constraint_coarse_range_host[2 * cell_id + 1] =
                      constraint_coarse_range_host[2 * cell_id] + n_cons;

                    // todo: mpi
                    memcpy(&l_to_g_coarse_host[cell_id * padding_length *
                                               n_components],
                           lexicographic_dof_indices.data(),
                           dofs_per_cell *
                             sizeof(dealii::types::global_dof_index));

                    cell_id++;
                  }

                n_inner_faces++;
              }
            else
              {
                if (matrix_type == MatrixType::active_matrix &&
                    neighbor->has_children())
                  continue;

                if (neighbor < cell)
                  continue;

                n_inner_faces++;
              }
          }
      }
  }

  template <int dim, typename Number>
  template <typename CellFilter>
  void
  ReinitHelper<dim, Number>::get_face_data(CellFilter        &cell,
                                           unsigned int      &inner_face_id,
                                           unsigned int      &boundary_face_id,
                                           const unsigned int color)
  {
    auto fill_data = [&](auto &fe_value, auto obj_id) {
      if (update_flags_inner_faces & dealii::update_JxW_values)
        {
          std::vector<double> JxW_values = fe_value.get_JxW_values();
          const unsigned int  offset     = obj_id * face_padding_length;
          for (unsigned int i = 0; i < q_points_per_face; ++i)
            face_JxW_host[i + offset] = static_cast<Number>(JxW_values[i]);
        }

      if (update_flags_inner_faces & dealii::update_gradients)
        {
          const std::vector<dealii::DerivativeForm<1, dim, dim>>
            &inv_jacobians = fe_value.get_inverse_jacobians();
          std::copy(
            &inv_jacobians[0][0][0],
            &inv_jacobians[0][0][0] +
              q_points_per_face * sizeof(dealii::DerivativeForm<1, dim, dim>) /
                sizeof(double),
            &face_inv_jacobian_host[obj_id * face_padding_length * dim * dim]);
        }

      if (update_flags_inner_faces & dealii::update_jacobians)
        {
          const std::vector<dealii::DerivativeForm<1, dim, dim>> &jacobians =
            fe_value.get_jacobians();
          std::copy(
            &jacobians[0][0][0],
            &jacobians[0][0][0] +
              q_points_per_face * sizeof(dealii::DerivativeForm<1, dim, dim>) /
                sizeof(double),
            &face_jacobian_host[obj_id * face_padding_length * dim * dim]);

          // determinant of inverse jacobian
          const unsigned int offset = obj_id * face_padding_length;
          const std::vector<dealii::DerivativeForm<1, dim, dim>>
            &inv_jacobians = fe_value.get_inverse_jacobians();
          for (unsigned int i = 0; i < q_points_per_face; ++i)
            face_inv_det_host[i + offset] =
              static_cast<Number>(inv_jacobians[i].determinant());

          // quadrature weights
          std::vector<double> quadrature_weights =
            fe_value.get_quadrature().get_weights();
          for (unsigned int i = 0; i < q_points_per_face; ++i)
            face_q_weights_host[i] = static_cast<Number>(quadrature_weights[i]);
        }

      if (update_flags_inner_faces & dealii::update_normal_vectors)
        {
          const std::vector<dealii::Tensor<1, dim>> &normal_vectors =
            fe_value.get_normal_vectors();
          std::copy(
            &normal_vectors[0][0],
            &normal_vectors[0][0] + q_points_per_face *
                                      sizeof(dealii::Tensor<1, dim>) /
                                      sizeof(double),
            &normal_vector_host[obj_id * face_padding_length * dim * 1]);
        }
    };

    for (const unsigned int face_no : cell->face_indices())
      {
        auto cell_info = std::make_pair<int, int>(cell->level(), cell->index());
        auto cell_id   = cell2cell_id[color][cell_info];

        if (cell->at_boundary(face_no))
          {
            boundary_face2cell_id_host[boundary_face_id]              = cell_id;
            face_number_host[n_inner_faces * 2 + boundary_face_id]    = face_no;
            subface_number_host[n_inner_faces * 2 + boundary_face_id] = -1;
            face_orientation_host[n_inner_faces * 2 + boundary_face_id] =
              cell->face_orientation(face_no);
            // todo: face_flip(), face_rotation()

            fe_face_values.reinit(cell, face_no);
            fill_data(fe_face_values, n_inner_faces * 2 + boundary_face_id);

            boundary_face_id++;
          }
        else
          {
            auto neighbor = cell->neighbor_or_periodic_neighbor(face_no);

            if (cell->neighbor_is_coarser(face_no))
              {
                const std::pair<unsigned int, unsigned int> neighbor_face_no =
                  cell->neighbor_of_coarser_neighbor(face_no);

                auto neighbor_info =
                  std::make_pair<int, int>(neighbor->level(),
                                           neighbor->index());
                int cell_id1;

                if (matrix_type == MatrixType::edge_up_matrix ||
                    matrix_type == MatrixType::edge_down_matrix ||
                    matrix_type == MatrixType::interface_matrix)
                  cell_id1 = cell2cell_id_coarse[color][neighbor_info];
                else
                  cell_id1 = cell2cell_id[color][neighbor_info];

                inner_face2cell_id_host[inner_face_id] = cell_id;
                face_number_host[inner_face_id]        = face_no;
                subface_number_host[inner_face_id]     = -2;
                face_orientation_host[inner_face_id] =
                  cell->face_orientation(face_no);

                inner_face2cell_id_host[inner_face_id + n_inner_faces] =
                  cell_id1;
                face_number_host[inner_face_id + n_inner_faces] =
                  neighbor_face_no.first;
                subface_number_host[inner_face_id + n_inner_faces] =
                  neighbor_face_no.second;
                face_orientation_host[inner_face_id + n_inner_faces] =
                  neighbor->face_orientation(neighbor_face_no.first);

                fe_face_values.reinit(cell, face_no);
                fill_data(fe_face_values, inner_face_id);

                fe_subface_values.reinit(neighbor,
                                         neighbor_face_no.first,
                                         neighbor_face_no.second);
                fill_data(fe_subface_values, inner_face_id + n_inner_faces);

                inner_face_id++;
              }
            else
              {
                if (matrix_type == MatrixType::active_matrix &&
                    neighbor->has_children())
                  continue;

                if (neighbor < cell)
                  continue;

                auto neighbor_info =
                  std::make_pair<int, int>(neighbor->level(),
                                           neighbor->index());
                auto cell_id1         = cell2cell_id[color][neighbor_info];
                auto neighbor_face_no = cell->neighbor_face_no(face_no);

                inner_face2cell_id_host[inner_face_id] = cell_id;
                face_number_host[inner_face_id]        = face_no;
                subface_number_host[inner_face_id]     = -1;
                face_orientation_host[inner_face_id] =
                  cell->face_orientation(face_no);

                inner_face2cell_id_host[inner_face_id + n_inner_faces] =
                  cell_id1;
                face_number_host[inner_face_id + n_inner_faces] =
                  neighbor_face_no;
                subface_number_host[inner_face_id + n_inner_faces] = -1;
                face_orientation_host[inner_face_id + n_inner_faces] =
                  neighbor->face_orientation(neighbor_face_no);

                fe_face_values.reinit(cell, face_no);
                fill_data(fe_face_values, inner_face_id);

                fe_face_values.reinit(neighbor, neighbor_face_no);
                fill_data(fe_face_values, inner_face_id + n_inner_faces);

                inner_face_id++;
              }
          }
      }
  }

  template <int dim, typename Number>
  void
  ReinitHelper<dim, Number>::alloc_and_copy_arrays(const unsigned int color)
  {
    const unsigned int n_cells = data->n_cells[color];

    unsigned int nn_cells = n_cells;
    if (matrix_type == MatrixType::edge_up_matrix ||
        matrix_type == MatrixType::edge_down_matrix)
      nn_cells *= 1 << dim - 1;
    else if (matrix_type == MatrixType::interface_matrix)
      nn_cells *= 1 << dim;

    // Local-to-global mapping
    alloc_and_copy(&data->local_to_global[color],
                   dealii::ArrayView<const dealii::types::global_dof_index>(
                     local_to_global_host.data(), local_to_global_host.size()),
                   nn_cells * padding_length * n_components);

    // Local-to-global mapping
    alloc_and_copy(&data->l_to_g_coarse[color],
                   dealii::ArrayView<const dealii::types::global_dof_index>(
                     l_to_g_coarse_host.data(), l_to_g_coarse_host.size()),
                   nn_cells * padding_length * n_components);

    // Quadrature points
    if (update_flags & dealii::update_quadrature_points)
      {
        alloc_and_copy(&data->q_points[color],
                       dealii::ArrayView<const dealii::Point<dim, Number>>(
                         q_points_host.data(), q_points_host.size()),
                       n_cells * padding_length);
      }

    // Jacobian determinants/quadrature weights
    if (update_flags & dealii::update_JxW_values)
      {
        alloc_and_copy(&data->JxW[color],
                       dealii::ArrayView<const Number>(JxW_host.data(),
                                                       JxW_host.size()),
                       n_cells * padding_length);
      }

    // Jacobians
    if (update_flags & dealii::update_jacobians)
      {
        // Reorder so that all J_11 elements are together, all J_12 elements
        // are together, etc., i.e., reorder indices from
        // cell_id*q_points_per_cell*dim*dim + q*dim*dim +i to
        // i*q_points_per_cell*n_cells + cell_id*q_points_per_cell+q
        transpose_in_place(jacobian_host, padding_length * n_cells, dim * dim);

        alloc_and_copy(&data->jacobian[color],
                       dealii::ArrayView<const Number>(jacobian_host.data(),
                                                       jacobian_host.size()),
                       n_cells * dim * dim * padding_length);

        alloc_and_copy(&data->inv_det[color],
                       dealii::ArrayView<const Number>(inv_det_host.data(),
                                                       inv_det_host.size()),
                       n_cells * padding_length);

        alloc_and_copy(&data->q_weights[color],
                       dealii::ArrayView<const Number>(q_weights_host.data(),
                                                       q_weights_host.size()),
                       padding_length);
      }

    // Inverse jacobians
    if (update_flags & dealii::update_gradients)
      {
        // Reorder so that all J_11 elements are together, all J_12 elements
        // are together, etc., i.e., reorder indices from
        // cell_id*q_points_per_cell*dim*dim + q*dim*dim +i to
        // i*q_points_per_cell*n_cells + cell_id*q_points_per_cell+q
        transpose_in_place(inv_jacobian_host,
                           padding_length * n_cells,
                           dim * dim);

        alloc_and_copy(&data->inv_jacobian[color],
                       dealii::ArrayView<const Number>(
                         inv_jacobian_host.data(), inv_jacobian_host.size()),
                       n_cells * dim * dim * padding_length);
      }

    // alloc_and_copy(
    //   &data->constraint_mask[color],
    //   dealii::ArrayView<
    //     const dealii::internal::MatrixFreeFunctions::ConstraintKinds>(
    //     constraint_mask_host.data(), constraint_mask_host.size()),
    //   n_cells * n_components);

    alloc_and_copy(&data->constraint_range[color],
                   dealii::ArrayView<const dealii::types::global_dof_index>(
                     constraint_range_host.data(),
                     constraint_range_host.size()),
                   n_cells * 2 * (1 << dim - 1));

    alloc_and_copy(&data->constraint_coarse_range[color],
                   dealii::ArrayView<const dealii::types::global_dof_index>(
                     constraint_coarse_range_host.data(),
                     constraint_coarse_range_host.size()),
                   n_cells * 2 * (1 << dim - 1));

    // for (auto c : constraint_coarse_range_host)
    //   std::cout << c << " ";
    // std::cout << std::endl;
  }

  template <int dim, typename Number>
  void
  ReinitHelper<dim, Number>::alloc_and_copy_face_arrays(
    const unsigned int color)
  {
    const unsigned int n_faces =
      data->n_inner_faces[color] * 2 + data->n_boundary_faces[color];

    alloc_and_copy(&data->face_number[color],
                   dealii::ArrayView<const dealii::types::global_dof_index>(
                     face_number_host.data(), face_number_host.size()),
                   n_faces);

    alloc_and_copy(&data->subface_number[color],
                   dealii::ArrayView<const int>(subface_number_host.data(),
                                                subface_number_host.size()),
                   n_faces);

    alloc_and_copy(&data->face_orientation[color],
                   dealii::ArrayView<const int>(face_orientation_host.data(),
                                                face_orientation_host.size()),
                   n_faces);

    alloc_and_copy(&data->inner_face2cell_id[color],
                   dealii::ArrayView<const dealii::types::global_dof_index>(
                     inner_face2cell_id_host.data(),
                     inner_face2cell_id_host.size()),
                   data->n_inner_faces[color] * 2);

    alloc_and_copy(&data->boundary_face2cell_id[color],
                   dealii::ArrayView<const dealii::types::global_dof_index>(
                     boundary_face2cell_id_host.data(),
                     boundary_face2cell_id_host.size()),
                   data->n_boundary_faces[color]);

    // Face jacobian determinants/quadrature weights
    if (update_flags_inner_faces & dealii::update_JxW_values)
      {
        alloc_and_copy(&data->face_JxW[color],
                       dealii::ArrayView<const Number>(face_JxW_host.data(),
                                                       face_JxW_host.size()),
                       n_faces * face_padding_length);
      }

    // face Jacobians
    if (update_flags_inner_faces & dealii::update_jacobians)
      {
        // Reorder so that all J_11 elements are together, all J_12 elements
        // are together, etc., i.e., reorder indices from
        // cell_id*q_points_per_cell*dim*dim + q*dim*dim +i to
        // i*q_points_per_cell*n_cells + cell_id*q_points_per_cell+q
        transpose_in_place(face_jacobian_host,
                           face_padding_length * n_faces,
                           dim * dim);

        alloc_and_copy(&data->face_jacobian[color],
                       dealii::ArrayView<const Number>(
                         face_jacobian_host.data(), face_jacobian_host.size()),
                       n_faces * dim * dim * face_padding_length);

        alloc_and_copy(&data->face_inv_det[color],
                       dealii::ArrayView<const Number>(
                         face_inv_det_host.data(), face_inv_det_host.size()),
                       n_faces * face_padding_length);

        alloc_and_copy(
          &data->face_q_weights[color],
          dealii::ArrayView<const Number>(face_q_weights_host.data(),
                                          face_q_weights_host.size()),
          face_padding_length);
      }

    // face Inverse jacobians
    if (update_flags_inner_faces & dealii::update_gradients)
      {
        // Reorder so that all J_11 elements are together, all J_12 elements
        // are together, etc., i.e., reorder indices from
        // cell_id*q_points_per_cell*dim*dim + q*dim*dim +i to
        // i*q_points_per_cell*n_cells + cell_id*q_points_per_cell+q
        transpose_in_place(face_inv_jacobian_host,
                           n_faces * face_padding_length,
                           dim * dim);

        alloc_and_copy(
          &data->face_inv_jacobian[color],
          dealii::ArrayView<const Number>(face_inv_jacobian_host.data(),
                                          face_inv_jacobian_host.size()),
          n_faces * dim * dim * face_padding_length);
      }

    // face normal vectors
    if (update_flags_inner_faces & dealii::update_normal_vectors)
      {
        transpose_in_place(normal_vector_host,
                           n_faces * face_padding_length,
                           dim * 1);

        alloc_and_copy(&data->normal_vector[color],
                       dealii::ArrayView<const Number>(
                         normal_vector_host.data(), normal_vector_host.size()),
                       n_faces * dim * 1 * face_padding_length);
      }
  }

  template <int dim, typename number, typename CellFilter>
  std::vector<dealii::types::global_dof_index>
  get_conflict_indices(const CellFilter                        &cell,
                       const dealii::AffineConstraints<number> &constraints)
  {
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      cell->get_fe().n_dofs_per_cell());
    cell->get_dof_indices(local_dof_indices);
    constraints.resolve_indices(local_dof_indices);

    return local_dof_indices;
  }



  template <typename Number>
  __global__ void
  copy_constrained_dofs(const dealii::types::global_dof_index *constrained_dofs,
                        const unsigned int n_constrained_dofs,
                        const unsigned int size,
                        const Number      *src,
                        Number            *dst)
  {
    const unsigned int dof =
      threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
    // When working with distributed vectors, the constrained dofs are
    // computed for ghosted vectors but we want to copy the values of the
    // constrained dofs of non-ghosted vectors.
    if ((dof < n_constrained_dofs) && (constrained_dofs[dof] < size))
      dst[constrained_dofs[dof]] = src[constrained_dofs[dof]];
  }



  template <typename Number>
  __global__ void
  set_constrained_dofs(const dealii::types::global_dof_index *constrained_dofs,
                       const unsigned int n_constrained_dofs,
                       const unsigned int size,
                       Number             val,
                       Number            *dst)
  {
    const unsigned int dof =
      threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
    // When working with distributed vectors, the constrained dofs are
    // computed for ghosted vectors but we want to set the values of the
    // constrained dofs of non-ghosted vectors.
    if ((dof < n_constrained_dofs) && (constrained_dofs[dof] < size))
      dst[constrained_dofs[dof]] = val;
  }

  template <int dim, typename Number, typename Functor>
  __global__ void
  apply_kernel_shmem(Functor            func,
                     const unsigned int color,
                     const Number      *src,
                     Number            *dst)
  {
    constexpr unsigned int cells_per_block = Functor::cells_per_block;

    constexpr unsigned int n_dofs_per_block =
      cells_per_block * Functor::n_local_dofs;
    constexpr unsigned int n_q_points_per_block =
      cells_per_block * Functor::n_q_points;
    // TODO make use of dynamically allocated shared memory
    __shared__ Number values[n_dofs_per_block];
    __shared__ Number gradients[dim][n_q_points_per_block];

    const unsigned int local_cell = threadIdx.x / Functor::n_dofs_1d;
    const unsigned int cell       = local_cell + cells_per_block * blockIdx.x;

    Number *gq[dim];
    for (unsigned int d = 0; d < dim; ++d)
      gq[d] = &gradients[d][local_cell * Functor::n_q_points];

    SharedData<dim, Number> shared_data(
      &values[local_cell * Functor::n_local_dofs], gq);

    // if (cell < gpu_data.n_cells) // todo should be n_cells or n_faces
    func(cell, color, &shared_data, src, dst);
  }



  template <int dim, typename Number, typename Functor>
  __global__ void
  evaluate_coeff(Functor                                      func,
                 const typename MatrixFree<dim, Number>::Data gpu_data)
  {
    constexpr unsigned int cells_per_block =
      cells_per_block_shmem(dim, Functor::n_dofs_1d - 1);

    const unsigned int local_cell = threadIdx.x / Functor::n_dofs_1d;
    const unsigned int cell =
      local_cell + cells_per_block * (blockIdx.x + gridDim.x * blockIdx.y);

    if (cell < gpu_data.n_cells)
      func(cell, &gpu_data);
  }


  template <int dim, typename Number>
  MatrixFree<dim, Number>::MatrixFree()
    : n_dofs(0)
    , constrained_dofs(nullptr)
    , padding_length(0)
    , my_id(-1)
    , dof_handler(nullptr)
  {}



  template <int dim, typename Number>
  MatrixFree<dim, Number>::~MatrixFree()
  {
    free();
  }


  template <int dim, typename Number>
  void
  MatrixFree<dim, Number>::free()
  {
    for (auto &q_points_color_ptr : q_points)
      dealii::Utilities::CUDA::free(q_points_color_ptr);
    q_points.clear();

    for (auto &local_to_global_color_ptr : local_to_global)
      dealii::Utilities::CUDA::free(local_to_global_color_ptr);
    local_to_global.clear();

    for (auto &inv_jacobian_color_ptr : inv_jacobian)
      dealii::Utilities::CUDA::free(inv_jacobian_color_ptr);
    inv_jacobian.clear();

    for (auto &JxW_color_ptr : JxW)
      dealii::Utilities::CUDA::free(JxW_color_ptr);
    JxW.clear();

    // for (auto &constraint_mask_color_ptr : constraint_mask)
    //   dealii::Utilities::CUDA::free(constraint_mask_color_ptr);
    // constraint_mask.clear();

    dealii::Utilities::CUDA::free(constrained_dofs);

    used_objects[my_id].store(false);
    my_id = -1;
  }


  template <int dim, typename Number>
  template <typename IteratorFiltersType>
  void
  MatrixFree<dim, Number>::reinit(
    const dealii::Mapping<dim>              &mapping,
    const dealii::DoFHandler<dim>           &dof_handler,
    const dealii::AffineConstraints<Number> &constraints,
    const dealii::Quadrature<1>             &quad,
    const IteratorFiltersType               &iterator_filter,
    const AdditionalData                    &additional_data)
  {
    this->matrix_type = additional_data.matrix_type;

    if (matrix_type == MatrixType::active_matrix)
      {
        Assert(
          (std::is_same<IteratorFiltersType,
                        dealii::IteratorFilters::LocallyOwnedCell>::value),
          (dealii::ExcMessage(
            "IteratorFiltersType must be of type IteratorFilters::LocallyOwnedCell.")));

        const auto &triangulation = dof_handler.get_triangulation();
        if (const auto parallel_triangulation =
              dynamic_cast<const dealii::parallel::TriangulationBase<dim> *>(
                &triangulation))
          internal_reinit<0, ActiveCellFilter, IteratorFiltersType>(
            mapping,
            dof_handler,
            constraints,
            quad,
            iterator_filter,
            std::make_shared<const MPI_Comm>(
              parallel_triangulation->get_communicator()),
            additional_data);
        else
          internal_reinit<0, ActiveCellFilter, IteratorFiltersType>(
            mapping,
            dof_handler,
            constraints,
            quad,
            iterator_filter,
            nullptr,
            additional_data);
      }
    else
      {
        Assert(
          (std::is_same<IteratorFiltersType,
                        dealii::IteratorFilters::LocallyOwnedLevelCell>::value),
          (dealii::ExcMessage(
            "IteratorFiltersType must be of type IteratorFilters::LocallyOwnedLevelCell")));

        const auto &triangulation = dof_handler.get_triangulation();
        if (const auto parallel_triangulation =
              dynamic_cast<const dealii::parallel::TriangulationBase<dim> *>(
                &triangulation))
          internal_reinit<1, LevelCellFilter, IteratorFiltersType>(
            mapping,
            dof_handler,
            constraints,
            quad,
            iterator_filter,
            std::make_shared<const MPI_Comm>(
              parallel_triangulation->get_communicator()),
            additional_data);
        else
          internal_reinit<1, LevelCellFilter, IteratorFiltersType>(
            mapping,
            dof_handler,
            constraints,
            quad,
            iterator_filter,
            nullptr,
            additional_data);
      }
  }

  template <int dim, typename Number>
  MatrixFree<dim, Number>::Data
  MatrixFree<dim, Number>::get_data(unsigned int color) const
  {
    Data data_copy;
    if (q_points.size() > 0)
      data_copy.q_points = q_points[color];
    if (jacobian.size() > 0)
      data_copy.jacobian = jacobian[color];
    if (inv_jacobian.size() > 0)
      data_copy.inv_jacobian = inv_jacobian[color];
    if (JxW.size() > 0)
      data_copy.JxW = JxW[color];
    if (inv_det.size() > 0)
      data_copy.inv_det = inv_det[color];
    if (q_weights.size() > 0)
      data_copy.q_weights = q_weights[color];

    data_copy.local_to_global = local_to_global[color];
    data_copy.id              = my_id;
    data_copy.n_cells         = n_cells[color];
    data_copy.padding_length  = padding_length;
    data_copy.row_start       = row_start[color];
    data_copy.use_coloring    = use_coloring;
    data_copy.is_primitive    = is_primitive;

    data_copy.constraint_mask  = constraint_mask[color];
    data_copy.constraint_range = constraint_range[color];

    data_copy.hanging_nodes_constraint = hanging_nodes_constraint;
    data_copy.hanging_nodes_constraint_indicator =
      hanging_nodes_constraint_indicator;
    data_copy.hanging_nodes_constraint_weights =
      hanging_nodes_constraint_weights;

    return data_copy;
  }

  template <int dim, typename Number>
  template <bool is_boundary_face>
  MatrixFree<dim, Number>::Data
  MatrixFree<dim, Number>::get_face_data(unsigned int color) const
  {
    Data data_copy;

    const unsigned int shift =
      is_boundary_face * n_inner_faces[color] * 2 * face_padding_length;

    if (face_jacobian.size() > 0)
      data_copy.face_jacobian = face_jacobian[color] + shift;
    if (face_inv_jacobian.size() > 0)
      data_copy.face_inv_jacobian = face_inv_jacobian[color] + shift;
    if (face_JxW.size() > 0)
      data_copy.face_JxW = face_JxW[color] + shift;
    if (normal_vector.size() > 0)
      data_copy.normal_vector = normal_vector[color] + shift;
    if (face_inv_det.size() > 0)
      data_copy.face_inv_det = face_inv_det[color] + shift;
    if (face_q_weights.size() > 0)
      data_copy.face_q_weights = face_q_weights[color];

    if (matrix_type == MatrixType::active_matrix ||
        matrix_type == MatrixType::level_matrix ||
        matrix_type == MatrixType::interface_matrix)
      {
        data_copy.local_to_global = local_to_global[color];
        data_copy.l_to_g_coarse   = local_to_global[color];
      }
    else if (matrix_type == MatrixType::edge_down_matrix)
      {
        data_copy.local_to_global = local_to_global[color];
        data_copy.l_to_g_coarse   = l_to_g_coarse[color];
      }
    else if (matrix_type == MatrixType::edge_up_matrix)
      {
        data_copy.local_to_global = l_to_g_coarse[color];
        data_copy.l_to_g_coarse   = local_to_global[color];
      }

    data_copy.face_number =
      face_number[color] + is_boundary_face * n_inner_faces[color] * 2;
    data_copy.subface_number =
      subface_number[color] + is_boundary_face * n_inner_faces[color] * 2;
    data_copy.face_orientation =
      face_orientation[color] + is_boundary_face * n_inner_faces[color] * 2;
    data_copy.id                  = my_id;
    data_copy.mg_level            = mg_level;
    data_copy.padding_length      = padding_length;
    data_copy.face_padding_length = face_padding_length;
    data_copy.use_coloring        = use_coloring;
    data_copy.is_primitive        = is_primitive;

    data_copy.constraint_range        = constraint_range[color];
    data_copy.constraint_coarse_range = constraint_coarse_range[color];

    data_copy.hanging_nodes_constraint = hanging_nodes_constraint;
    data_copy.hanging_nodes_constraint_indicator =
      hanging_nodes_constraint_indicator;
    data_copy.hanging_nodes_constraint_weights =
      hanging_nodes_constraint_weights;

    data_copy.hanging_nodes_constraint_coarse = hanging_nodes_constraint_coarse;

    data_copy.face2cell_id = is_boundary_face ? boundary_face2cell_id[color] :
                                                inner_face2cell_id[color];
    data_copy.n_faces =
      is_boundary_face ? n_boundary_faces[color] : n_inner_faces[color];

    data_copy.n_cells = n_boundary_faces[color] + n_inner_faces[color] * 2;

    data_copy.matrix_type = matrix_type;

    return data_copy;
  }

  template <int dim, typename Number>
  void
  MatrixFree<dim, Number>::initialize_dof_vector(
    dealii::LinearAlgebra::distributed::Vector<Number,
                                               dealii::MemorySpace::CUDA> &vec)
    const
  {
    if (partitioner)
      vec.reinit(partitioner);
    else
      vec.reinit(n_dofs);
  }

  template <int dim, typename Number>
  unsigned int
  MatrixFree<dim, Number>::get_padding_length() const
  {
    return padding_length;
  }

  template <int dim, typename Number>
  unsigned int
  MatrixFree<dim, Number>::get_face_padding_length() const
  {
    return face_padding_length;
  }

  template <int dim, typename Number>
  template <typename Functor, typename VectorType>
  void
  MatrixFree<dim, Number>::cell_loop(const Functor    &func,
                                     const VectorType &src,
                                     VectorType       &dst) const
  {
    if (partitioner)
      distributed_cell_loop(func, src, dst);
    else
      serial_cell_loop(func, src, dst);
  }



  template <int dim, typename Number>
  template <typename Functor, typename VectorType>
  void
  MatrixFree<dim, Number>::inner_face_loop(const Functor    &func,
                                           const VectorType &src,
                                           VectorType       &dst) const
  {
    // Execute the loop on the boundary faces
    for (unsigned int i = 0; i < n_colors; ++i)
      if (n_inner_faces[i] > 0)
        {
          apply_kernel_shmem<dim, Number, Functor>
            <<<grid_dim_inner_face[i], block_dim_inner_face[i]>>>(
              func, i, src.get_values(), dst.get_values());
          AssertCudaKernel();
        }
  }



  template <int dim, typename Number>
  template <typename Functor, typename VectorType>
  void
  MatrixFree<dim, Number>::boundary_face_loop(const Functor    &func,
                                              const VectorType &src,
                                              VectorType       &dst) const
  {
    // Execute the loop on the boundary faces
    for (unsigned int i = 0; i < n_colors; ++i)
      if (n_boundary_faces[i] > 0)
        {
          apply_kernel_shmem<dim, Number, Functor>
            <<<grid_dim_boundary_face[i], block_dim_boundary_face[i]>>>(
              func, i, src.get_values(), dst.get_values());
          AssertCudaKernel();
        }
  }



  template <int dim, typename Number>
  template <typename Functor>
  void
  MatrixFree<dim, Number>::evaluate_coefficients(Functor func) const
  {
    for (unsigned int i = 0; i < n_colors; ++i)
      if (n_cells[i] > 0)
        {
          evaluate_coeff<dim, Number, Functor>
            <<<grid_dim[i], block_dim[i]>>>(func, get_data(i));
          AssertCudaKernel();
        }
  }

  template <int dim, typename Number>
  std::size_t
  MatrixFree<dim, Number>::memory_consumption() const
  {
    // todo: add face storage
    // First compute the size of n_cells, row_starts, kernel launch parameters,
    // and constrained_dofs
    std::size_t bytes = n_cells.size() * sizeof(unsigned int) * 2 +
                        2 * n_colors * sizeof(dim3) +
                        n_constrained_dofs * sizeof(unsigned int);

    // For each color, add local_to_global, inv_jacobian, JxW, and q_points.
    for (unsigned int i = 0; i < n_colors; ++i)
      {
        bytes += n_cells[i] * padding_length * sizeof(unsigned int) +
                 n_cells[i] * padding_length * dim * dim * sizeof(Number) +
                 n_cells[i] * padding_length * sizeof(Number) +
                 n_cells[i] * padding_length * sizeof(point_type) +
                 n_cells[i] * sizeof(unsigned int);
      }

    return bytes;
  }


  template <int dim, typename Number>
  template <bool is_LevelMG,
            typename CellIterator,
            typename IteratorFiltersType>
  void
  MatrixFree<dim, Number>::internal_reinit(
    const dealii::Mapping<dim>              &mapping,
    const dealii::DoFHandler<dim>           &dof_handler_,
    const dealii::AffineConstraints<Number> &constraints,
    const dealii::Quadrature<1>             &quad,
    const IteratorFiltersType               &iterator_filter,
    std::shared_ptr<const MPI_Comm>          comm,
    const AdditionalData                     additional_data)
  {
    dof_handler = &dof_handler_;

    if (typeid(Number) == typeid(double))
      cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    dealii::UpdateFlags update_flags = additional_data.mapping_update_flags;
    dealii::UpdateFlags update_flags_inner_faces =
      additional_data.mapping_update_flags_inner_faces;

    if (update_flags & dealii::update_gradients)
      update_flags |= dealii::update_JxW_values;

    if (update_flags_inner_faces & dealii::update_gradients)
      update_flags_inner_faces |= dealii::update_JxW_values;

    this->use_coloring = additional_data.use_coloring;
    this->overlap_communication_computation =
      additional_data.overlap_communication_computation;
    this->mg_level = additional_data.mg_level;
    this->my_id    = additional_data.my_id;

    // TODO: only free if we actually need arrays of different length
    // free();

    n_dofs = is_LevelMG ? dof_handler->n_dofs(mg_level) : dof_handler->n_dofs();

    const dealii::FiniteElement<dim> &fe = dof_handler->get_fe();

    fe_degree    = fe.degree;
    is_primitive = fe.is_primitive();

    // TODO this should be a templated parameter
    const unsigned int n_dofs_1d     = fe_degree + 1 + is_primitive;
    const unsigned int n_q_points_1d = quad.size();

    Assert(n_dofs_1d == n_q_points_1d,
           dealii::ExcMessage("n_q_points_1d must be equal to n_dofs_1d."));

    // Set padding length to the closest power of two larger than or equal to
    // the number of threads. But why padding?
    padding_length =
      1 << static_cast<unsigned int>(std::ceil(dim * std::log2(n_dofs_1d)));

    face_padding_length = 1 << static_cast<unsigned int>(
                            std::ceil((dim - 1) * std::log2(n_dofs_1d)));

    n_components      = fe.n_components();
    dofs_per_cell     = fe.n_dofs_per_cell();
    q_points_per_cell = std::pow(n_q_points_1d, dim);
    q_points_per_face = std::pow(n_q_points_1d, dim - 1);

    auto qpoints  = quad.get_points();
    auto qweights = quad.get_weights();

    for (auto &q : qpoints)
      q[0] = q[0] / 2;

    dealii::Quadrature<1> sub_quad0(qpoints, qweights);

    for (auto &q : qpoints)
      q[0] = q[0] + 0.5;

    dealii::Quadrature<1> sub_quad1(qpoints, qweights);

    const dealii::internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info(
      quad, fe);
    const dealii::internal::MatrixFreeFunctions::ShapeInfo<Number>
      sub_shape_info0(sub_quad0, fe);
    const dealii::internal::MatrixFreeFunctions::ShapeInfo<Number>
      sub_shape_info1(sub_quad1, fe);

    std::array<
      dealii::internal::MatrixFreeFunctions::UnivariateShapeData<Number>,
      2>
      shape_data;
    std::array<
      dealii::internal::MatrixFreeFunctions::UnivariateShapeData<Number>,
      2>
      sub_shape_data0;
    std::array<
      dealii::internal::MatrixFreeFunctions::UnivariateShapeData<Number>,
      2>
      sub_shape_data1;

    for (auto d = 0U; d < 2; ++d)
      {
        shape_data[d]      = shape_info.get_shape_data(d, 0);
        sub_shape_data0[d] = sub_shape_info0.get_shape_data(d, 0);
        sub_shape_data1[d] = sub_shape_info1.get_shape_data(d, 0);
      }
    // todo sub shape info for RT

    unsigned int size_shape_values = n_dofs_1d * n_q_points_1d * sizeof(Number);
    unsigned int n_shape_values    = n_dofs_1d * n_q_points_1d;

    dealii::FE_DGQArbitraryNodes<1> fe_quad_co(quad);
    const dealii::internal::MatrixFreeFunctions::ShapeInfo<Number>
      shape_info_co(quad, fe_quad_co);

    dealii::FE_DGQArbitraryNodes<1> fe_subquad_co0(sub_quad0);
    const dealii::internal::MatrixFreeFunctions::ShapeInfo<Number>
      sub_shape_info_co0(sub_quad0, fe_subquad_co0);

    dealii::FE_DGQArbitraryNodes<1> fe_subquad_co1(sub_quad1);
    const dealii::internal::MatrixFreeFunctions::ShapeInfo<Number>
      sub_shape_info_co1(sub_quad1, fe_subquad_co1);

    unsigned int size_co_shape_values =
      n_q_points_1d * n_q_points_1d * sizeof(Number);

    // // Check if we already a part of the constant memory allocated to us. If
    // // not, we try to get a block of memory.
    // bool found_id = false;
    // while (!found_id && (!is_LevelMG || my_id == -1))
    //   {
    //     ++my_id;
    //     Assert(
    //       my_id < static_cast<int>(mf_n_concurrent_objects),
    //       dealii::ExcMessage(
    //         "Maximum number of concurrent MatrixFree objects reached.
    //         Increase mf_n_concurrent_objects"));
    //     bool f   = false;
    //     found_id = used_objects[my_id].compare_exchange_strong(f, true);
    //   }

    // // todo
    // my_id = 0;

    std::vector<Number> empty(n_shape_values * 6, 0.0);

    // todo: should init all constant memory to avoid nan.

    cudaError_t cuda_error =
      cudaMemcpyToSymbol(get_cell_shape_values<Number>(0),
                         empty.data(),
                         size_shape_values * 6,
                         my_id * data_array_size * sizeof(Number),
                         cudaMemcpyHostToDevice);
    AssertCuda(cuda_error);

    cuda_error =
      cudaMemcpyToSymbol(get_cell_shape_values<Number>(
                           0), // todo check (0) or (my_id)
                         shape_data[0].shape_values.data(),
                         shape_data[0].shape_values.size() * sizeof(Number),
                         my_id * data_array_size * sizeof(Number),
                         cudaMemcpyHostToDevice);
    AssertCuda(cuda_error);

    cuda_error = cudaMemcpyToSymbol(get_cell_shape_values<Number>(0),
                                    sub_shape_data0[0].shape_values.data(),
                                    sub_shape_data0[0].shape_values.size() *
                                      sizeof(Number),
                                    (my_id * data_array_size + n_shape_values) *
                                      sizeof(Number),
                                    cudaMemcpyHostToDevice);
    AssertCuda(cuda_error);

    cuda_error = cudaMemcpyToSymbol(
      get_cell_shape_values<Number>(0),
      sub_shape_data1[0].shape_values.data(),
      sub_shape_data1[0].shape_values.size() * sizeof(Number),
      (my_id * data_array_size + n_shape_values * 2) * sizeof(Number),
      cudaMemcpyHostToDevice);
    AssertCuda(cuda_error);

    if (!is_primitive)
      {
        cuda_error =
          cudaMemcpyToSymbol(get_cell_shape_values<Number>(0),
                             shape_data[1].shape_values.data(),
                             shape_data[1].shape_values.size() * sizeof(Number),
                             (my_id * data_array_size + n_shape_values * 3) *
                               sizeof(Number),
                             cudaMemcpyHostToDevice);
        AssertCuda(cuda_error);

        cuda_error = cudaMemcpyToSymbol(
          get_cell_shape_values<Number>(0),
          sub_shape_data0[1].shape_values.data(),
          sub_shape_data0[1].shape_values.size() * sizeof(Number),
          (my_id * data_array_size + n_shape_values * 4) * sizeof(Number),
          cudaMemcpyHostToDevice);
        AssertCuda(cuda_error);

        cuda_error = cudaMemcpyToSymbol(
          get_cell_shape_values<Number>(0),
          sub_shape_data1[1].shape_values.data(),
          sub_shape_data1[1].shape_values.size() * sizeof(Number),
          (my_id * data_array_size + n_shape_values * 5) * sizeof(Number),
          cudaMemcpyHostToDevice);
        AssertCuda(cuda_error);
      }

    if (update_flags & dealii::update_gradients)
      {
        cuda_error =
          cudaMemcpyToSymbol(get_cell_shape_gradients<Number>(0),
                             empty.data(),
                             size_shape_values * 6,
                             my_id * data_array_size * sizeof(Number),
                             cudaMemcpyHostToDevice);
        AssertCuda(cuda_error);

        cuda_error =
          cudaMemcpyToSymbol(get_cell_shape_gradients<Number>(0),
                             shape_data[0].shape_gradients.data(),
                             shape_data[0].shape_gradients.size() *
                               sizeof(Number),
                             my_id * data_array_size * sizeof(Number),
                             cudaMemcpyHostToDevice);
        AssertCuda(cuda_error);

        cuda_error = cudaMemcpyToSymbol(
          get_cell_shape_gradients<Number>(0),
          sub_shape_data0[0].shape_gradients.data(),
          sub_shape_data0[0].shape_gradients.size() * sizeof(Number),
          (my_id * data_array_size + n_shape_values) * sizeof(Number),
          cudaMemcpyHostToDevice);
        AssertCuda(cuda_error);

        cuda_error = cudaMemcpyToSymbol(
          get_cell_shape_gradients<Number>(0),
          sub_shape_data1[0].shape_gradients.data(),
          sub_shape_data1[0].shape_gradients.size() * sizeof(Number),
          (my_id * data_array_size + n_shape_values * 2) * sizeof(Number),
          cudaMemcpyHostToDevice);
        AssertCuda(cuda_error);

        if (!is_primitive)
          {
            cuda_error = cudaMemcpyToSymbol(
              get_cell_shape_gradients<Number>(0),
              shape_data[1].shape_gradients.data(),
              shape_data[1].shape_gradients.size() * sizeof(Number),
              (my_id * data_array_size + n_shape_values * 3) * sizeof(Number),
              cudaMemcpyHostToDevice);
            AssertCuda(cuda_error);

            cuda_error = cudaMemcpyToSymbol(
              get_cell_shape_gradients<Number>(0),
              sub_shape_data0[1].shape_gradients.data(),
              sub_shape_data0[1].shape_gradients.size() * sizeof(Number),
              (my_id * data_array_size + n_shape_values * 4) * sizeof(Number),
              cudaMemcpyHostToDevice);
            AssertCuda(cuda_error);

            cuda_error = cudaMemcpyToSymbol(
              get_cell_shape_gradients<Number>(0),
              sub_shape_data1[1].shape_gradients.data(),
              sub_shape_data1[1].shape_gradients.size() * sizeof(Number),
              (my_id * data_array_size + n_shape_values * 5) * sizeof(Number),
              cudaMemcpyHostToDevice);
            AssertCuda(cuda_error);
          }

        cuda_error =
          cudaMemcpyToSymbol(get_cell_co_shape_gradients<Number>(0),
                             shape_info_co.data.front().shape_gradients.data(),
                             size_co_shape_values,
                             my_id * data_array_size * sizeof(Number),
                             cudaMemcpyHostToDevice);
        AssertCuda(cuda_error);

        cuda_error = cudaMemcpyToSymbol(
          get_cell_co_shape_gradients<Number>(0),
          sub_shape_info_co0.data.front().shape_gradients.data(),
          size_co_shape_values,
          (my_id * data_array_size + n_shape_values) * sizeof(Number),
          cudaMemcpyHostToDevice);
        AssertCuda(cuda_error);

        cuda_error = cudaMemcpyToSymbol(
          get_cell_co_shape_gradients<Number>(0),
          sub_shape_info_co1.data.front().shape_gradients.data(),
          size_co_shape_values,
          (my_id * data_array_size + n_shape_values * 2) * sizeof(Number),
          cudaMemcpyHostToDevice);
        AssertCuda(cuda_error);
      }

    // todo: use less constant memory
    if (update_flags_inner_faces & dealii::update_values)
      {
        std::vector<Number> face_shape_value(2 * n_shape_values, 0.0);
        // face_shape_value.resize(2 * n_shape_values);
        // point 0
        auto shape_value_on_face0 =
          shape_data[!is_primitive].shape_data_on_face[0];
        for (unsigned int i = 0; i < n_q_points_1d; ++i)
          face_shape_value[i * n_q_points_1d] = shape_value_on_face0[i];
        // point 1
        auto shape_value_on_face1 =
          shape_data[!is_primitive].shape_data_on_face[1];
        for (unsigned int i = 0; i < n_q_points_1d; ++i)
          face_shape_value[(i + n_q_points_1d) * n_q_points_1d] =
            shape_value_on_face1[i];

        cuda_error =
          cudaMemcpyToSymbol(get_face_shape_values<Number>(0),
                             empty.data(),
                             size_shape_values * 6,
                             my_id * data_array_size * sizeof(Number),
                             cudaMemcpyHostToDevice);
        AssertCuda(cuda_error);

        cuda_error =
          cudaMemcpyToSymbol(get_face_shape_values<Number>(0),
                             face_shape_value.data(),
                             size_shape_values * 2,
                             my_id * data_array_size * sizeof(Number),
                             cudaMemcpyHostToDevice);
        AssertCuda(cuda_error);
      }

    if (update_flags_inner_faces & dealii::update_gradients)
      {
        std::vector<Number> face_shape_gradients(2 * n_shape_values, 0.0);
        // face_shape_gradients.resize(2 * n_shape_values);
        // point 0
        auto shape_gradients_on_face0 =
          shape_data[!is_primitive].shape_data_on_face[0];
        for (unsigned int i = 0; i < n_q_points_1d - 1; ++i)
          face_shape_gradients[i * n_q_points_1d] =
            shape_gradients_on_face0[i + n_q_points_1d - !is_primitive];
        // point 1
        auto shape_gradients_on_face1 =
          shape_data[!is_primitive].shape_data_on_face[1];
        for (unsigned int i = 0; i < n_q_points_1d - 1; ++i)
          face_shape_gradients[(i + n_q_points_1d) * n_q_points_1d] =
            shape_gradients_on_face1[i + n_q_points_1d - !is_primitive];

        cuda_error =
          cudaMemcpyToSymbol(get_face_shape_gradients<Number>(0),
                             empty.data(),
                             size_shape_values * 6,
                             my_id * data_array_size * sizeof(Number),
                             cudaMemcpyHostToDevice);
        AssertCuda(cuda_error);

        cuda_error =
          cudaMemcpyToSymbol(get_face_shape_gradients<Number>(0),
                             face_shape_gradients.data(),
                             size_shape_values * 2,
                             my_id * data_array_size * sizeof(Number),
                             cudaMemcpyHostToDevice);
        AssertCuda(cuda_error);


        std::vector<Number> face_co_shape_gradients;
        face_co_shape_gradients.resize(2 * n_shape_values);
        // point 0
        auto co_shape_gradients_on_face0 =
          shape_info_co.data.front().shape_data_on_face[0].data();
        for (unsigned int i = 0; i < n_q_points_1d; ++i)
          face_co_shape_gradients[i * n_q_points_1d] =
            co_shape_gradients_on_face0[i + n_q_points_1d];
        // point 1
        auto co_shape_gradients_on_face1 =
          shape_info_co.data.front().shape_data_on_face[1].data();
        for (unsigned int i = 0; i < n_q_points_1d; ++i)
          face_co_shape_gradients[(i + n_q_points_1d) * n_q_points_1d] =
            co_shape_gradients_on_face1[i + n_q_points_1d];


        cuda_error =
          cudaMemcpyToSymbol(get_face_co_shape_gradients<Number>(0),
                             face_co_shape_gradients.data(),
                             size_shape_values * 2,
                             my_id * data_array_size * sizeof(Number),
                             cudaMemcpyHostToDevice);
        AssertCuda(cuda_error);
      }

    // Setup the number of cells per CUDA thread block
    cells_per_block          = cells_per_block_shmem(dim, fe_degree);
    inner_faces_per_block    = 1; // todo
    boundary_faces_per_block = 1; // todo

    ReinitHelper<dim, Number> helper(this,
                                     mapping,
                                     fe,
                                     quad,
                                     shape_info,
                                     *dof_handler,
                                     constraints,
                                     update_flags,
                                     update_flags_inner_faces);

    typename dealii::DoFHandler<dim>::cell_iterator beginc;
    typename dealii::DoFHandler<dim>::cell_iterator endc;

    if (is_LevelMG)
      {
        beginc = dof_handler->begin_mg(mg_level);
        endc   = dof_handler->end_mg(mg_level);
      }
    else
      {
        beginc = dof_handler->begin_active();
        endc   = dof_handler->end();
      }

    // Create a graph coloring
    CellIterator begin(iterator_filter, beginc);
    CellIterator end(iterator_filter, endc);

    std::vector<std::vector<CellIterator>> graph;

    if (begin != end)
      {
        if (additional_data.use_coloring)
          {
            const auto fun = [&](const CellIterator &filter) {
              return get_conflict_indices<dim, Number>(filter, constraints);
            };
            graph = dealii::GraphColoring::make_graph_coloring(begin, end, fun);
          }
        else
          {
            graph.clear();
            if (additional_data.overlap_communication_computation)
              {
                // We create one color (1) with the cells on the boundary of the
                // local domain and two colors (0 and 2) with the interior
                // cells.
                graph.resize(3, std::vector<CellIterator>());

                std::vector<bool> ghost_vertices(
                  dof_handler->get_triangulation().n_vertices(), false);

                // todo: fix loop range
                for (const auto &cell :
                     dof_handler->get_triangulation().active_cell_iterators())
                  if (cell->is_ghost())
                    for (unsigned int i = 0;
                         i < dealii::GeometryInfo<dim>::vertices_per_cell;
                         i++)
                      ghost_vertices[cell->vertex_index(i)] = true;

                std::vector<CellIterator> inner_cells;

                for (auto cell = begin; cell != end; ++cell)
                  {
                    bool ghost_vertex = false;

                    for (unsigned int i = 0;
                         i < dealii::GeometryInfo<dim>::vertices_per_cell;
                         i++)
                      if (ghost_vertices[cell->vertex_index(i)])
                        {
                          ghost_vertex = true;
                          break;
                        }

                    if (ghost_vertex)
                      graph[1].emplace_back(cell);
                    else
                      inner_cells.emplace_back(cell);
                  }
                for (unsigned i = 0; i < inner_cells.size(); ++i)
                  if (i < inner_cells.size() / 2)
                    graph[0].emplace_back(inner_cells[i]);
                  else
                    graph[2].emplace_back(inner_cells[i]);
              }
            else
              {
                // If we are not using coloring, all the cells belong to the
                // same color.
                graph.resize(1, std::vector<CellIterator>());
                for (auto cell = begin; cell != end; ++cell)
                  graph[0].emplace_back(cell);
              }
          }
      }
    n_colors = graph.size();

    helper.setup_color_arrays(n_colors);

    dealii::IndexSet locally_owned_dofs;
    dealii::IndexSet locally_relevant_dofs;
    if (comm)
      {
        locally_relevant_dofs =
          is_LevelMG ?
            dealii::DoFTools::extract_locally_relevant_level_dofs(*dof_handler,
                                                                  mg_level) :
            dealii::DoFTools::extract_locally_relevant_dofs(*dof_handler);
        locally_owned_dofs = is_LevelMG ?
                               dof_handler->locally_owned_mg_dofs(mg_level) :
                               dof_handler->locally_owned_dofs();
        partitioner = std::make_shared<dealii::Utilities::MPI::Partitioner>(
          locally_owned_dofs, locally_relevant_dofs, *comm);
      }

    for (unsigned int i = 0; i < n_colors; ++i)
      {
        helper.n_inner_faces    = 0;
        helper.n_boundary_faces = 0;

        n_cells[i] = graph[i].size();
        helper.setup_cell_arrays(i);
        typename std::vector<CellIterator>::iterator cell = graph[i].begin(),
                                                     end_cell = graph[i].end();
        for (unsigned int cell_id = 0; cell != end_cell; ++cell)
          {
            helper.get_cell_data(*cell, cell_id, partitioner, i);
          }

        n_inner_faces[i]    = helper.n_inner_faces;
        n_boundary_faces[i] = helper.n_boundary_faces;

        helper.alloc_and_copy_arrays(i);
      }

    // Setup faces
    for (unsigned int i = 0; i < n_colors; ++i)
      {
        unsigned int inner_face_id    = 0;
        unsigned int boundary_face_id = 0;

        helper.setup_face_arrays(i);
        typename std::vector<CellIterator>::iterator cell = graph[i].begin(),
                                                     end_cell = graph[i].end();
        for (unsigned int cell_id = 0; cell != end_cell; ++cell, ++cell_id)
          helper.get_face_data(*cell, inner_face_id, boundary_face_id, i);

        helper.alloc_and_copy_face_arrays(i);
      }

    // Setup row starts
    if (n_colors > 0)
      row_start[0] = 0;
    for (unsigned int i = 1; i < n_colors; ++i)
      row_start[i] = row_start[i - 1] + n_cells[i - 1] * get_padding_length();


    // Constrained indices
    n_constrained_dofs = constraints.n_constraints();

    if (n_constrained_dofs != 0)
      {
        // hanging node constraints
        std::vector<dealii::types::global_dof_index> h_c;
        std::vector<int>                             h_c_i;
        std::vector<Number>                          h_c_w;

        std::vector<dealii::types::global_dof_index> h_c_coarse;


        auto lines = constraints.get_lines();
        for (auto &l : lines)
          if (l.entries.size() > 0)
            for (auto e : l.entries)
              {
                if (e.second < -100)
                  h_c_coarse.push_back(e.first);
                else
                  {
                    h_c.push_back(l.index);
                    h_c_i.push_back(e.first);
                    h_c_w.push_back(e.second);
                  }
              }
          else
            {
              h_c.push_back(l.index);
              h_c_i.push_back(0);
              h_c_w.push_back(0);
            }

        // for (auto h : h_c_coarse)
        //   std::cout << h << " ";
        // std::cout << std::endl << std::endl;

        alloc_and_copy(&hanging_nodes_constraint,
                       dealii::ArrayView<const dealii::types::global_dof_index>(
                         h_c.data(), h_c.size()),
                       h_c.size());

        alloc_and_copy(&hanging_nodes_constraint_indicator,
                       dealii::ArrayView<const int>(h_c_i.data(), h_c_i.size()),
                       h_c_i.size());

        alloc_and_copy(&hanging_nodes_constraint_weights,
                       dealii::ArrayView<const Number>(h_c_w.data(),
                                                       h_c_w.size()),
                       h_c_w.size());


        alloc_and_copy(&hanging_nodes_constraint_coarse,
                       dealii::ArrayView<const dealii::types::global_dof_index>(
                         h_c_coarse.data(), h_c_coarse.size()),
                       h_c_coarse.size());

        const unsigned int constraint_n_blocks =
          std::ceil(static_cast<double>(n_constrained_dofs) /
                    static_cast<double>(block_size));
        const unsigned int constraint_x_n_blocks =
          std::round(std::sqrt(constraint_n_blocks));
        const unsigned int constraint_y_n_blocks =
          std::ceil(static_cast<double>(constraint_n_blocks) /
                    static_cast<double>(constraint_x_n_blocks));

        constraint_grid_dim =
          dim3(constraint_x_n_blocks, constraint_y_n_blocks);
        constraint_block_dim = dim3(block_size);

        std::vector<dealii::types::global_dof_index> constrained_dofs_host(
          n_constrained_dofs);

        if (partitioner)
          {
            const unsigned int n_local_dofs =
              locally_relevant_dofs.n_elements();
            unsigned int i_constraint = 0;
            for (unsigned int i = 0; i < n_local_dofs; ++i)
              {
                // is_constrained uses a global dof id but
                // constrained_dofs_host works on the local id
                if (constraints.is_constrained(partitioner->local_to_global(i)))
                  {
                    constrained_dofs_host[i_constraint] = i;
                    ++i_constraint;
                  }
              }
          }
        else
          {
            const unsigned int n_local_dofs = is_LevelMG ?
                                                dof_handler->n_dofs(mg_level) :
                                                dof_handler->n_dofs();
            unsigned int       i_constraint = 0;
            for (unsigned int i = 0; i < n_local_dofs; ++i)
              {
                if (constraints.is_constrained(i))
                  {
                    constrained_dofs_host[i_constraint] = i;
                    ++i_constraint;
                  }
              }
          }

        cuda_error = cudaMalloc(&constrained_dofs,
                                n_constrained_dofs *
                                  sizeof(dealii::types::global_dof_index));
        AssertCuda(cuda_error);

        cuda_error = cudaMemcpy(constrained_dofs,
                                constrained_dofs_host.data(),
                                n_constrained_dofs *
                                  sizeof(dealii::types::global_dof_index),
                                cudaMemcpyHostToDevice);
        AssertCuda(cuda_error);
      }
  }



  template <int dim, typename Number>
  template <typename Functor, typename VectorType>
  void
  MatrixFree<dim, Number>::serial_cell_loop(const Functor    &func,
                                            const VectorType &src,
                                            VectorType       &dst) const
  {
    // Execute the loop on the cells
    for (unsigned int i = 0; i < n_colors; ++i)
      if (n_cells[i] > 0)
        {
          apply_kernel_shmem<dim, Number, Functor>
            <<<grid_dim[i], block_dim[i]>>>(func,
                                            i,
                                            src.get_values(),
                                            dst.get_values());
          AssertCudaKernel();
        }
  }



  template <int dim, typename Number>
  template <typename Functor>
  void
  MatrixFree<dim, Number>::distributed_cell_loop(
    const Functor &func,
    const dealii::LinearAlgebra::distributed::Vector<Number,
                                                     dealii::MemorySpace::CUDA>
                                                                          &src,
    dealii::LinearAlgebra::distributed::Vector<Number,
                                               dealii::MemorySpace::CUDA> &dst)
    const
  {
    // in case we have compatible partitioners, we can simply use the provided
    // vectors
    if (src.get_partitioner().get() == partitioner.get() &&
        dst.get_partitioner().get() == partitioner.get())
      {
        // This code is inspired to the code in TaskInfo::loop.
        if (overlap_communication_computation)
          {
            src.update_ghost_values_start(0);
            // In parallel, it's possible that some processors do not own any
            // cells.
            if (n_cells[0] > 0)
              {
                apply_kernel_shmem<dim, Number, Functor>
                  <<<grid_dim[0], block_dim[0]>>>(func,
                                                  0,
                                                  src.get_values(),
                                                  dst.get_values());
                AssertCudaKernel();
              }
            src.update_ghost_values_finish();

            // In serial this color does not exist because there are no ghost
            // cells
            if (n_cells[1] > 0)
              {
                apply_kernel_shmem<dim, Number, Functor>
                  <<<grid_dim[1], block_dim[1]>>>(func,
                                                  1,
                                                  src.get_values(),
                                                  dst.get_values());
                AssertCudaKernel();
                // We need a synchronization point because we don't want
                // CUDA-aware MPI to start the MPI communication until the
                // kernel is done.
                cudaDeviceSynchronize();
              }

            dst.compress_start(0, dealii::VectorOperation::add);
            // When the mesh is coarse it is possible that some processors do
            // not own any cells
            if (n_cells[2] > 0)
              {
                apply_kernel_shmem<dim, Number, Functor>
                  <<<grid_dim[2], block_dim[2]>>>(func,
                                                  2,
                                                  src.get_values(),
                                                  dst.get_values());
                AssertCudaKernel();
              }
            dst.compress_finish(dealii::VectorOperation::add);
          }
        else
          {
            src.update_ghost_values();

            // Execute the loop on the cells
            for (unsigned int i = 0; i < n_colors; ++i)
              if (n_cells[i] > 0)
                {
                  apply_kernel_shmem<dim, Number, Functor>
                    <<<grid_dim[i], block_dim[i]>>>(func,
                                                    i,
                                                    src.get_values(),
                                                    dst.get_values());
                }
            dst.compress(dealii::VectorOperation::add);
          }
        src.zero_out_ghost_values();
      }
    else
      {
        // Create the ghosted source and the ghosted destination
        dealii::LinearAlgebra::distributed::Vector<Number,
                                                   dealii::MemorySpace::CUDA>
          ghosted_src(partitioner);
        dealii::LinearAlgebra::distributed::Vector<Number,
                                                   dealii::MemorySpace::CUDA>
          ghosted_dst(ghosted_src);
        ghosted_src = src;
        ghosted_dst = dst;

        // Execute the loop on the cells
        for (unsigned int i = 0; i < n_colors; ++i)
          if (n_cells[i] > 0)
            {
              apply_kernel_shmem<dim, Number, Functor>
                <<<grid_dim[i], block_dim[i]>>>(func,
                                                i,
                                                ghosted_src.get_values(),
                                                ghosted_dst.get_values());
              AssertCudaKernel();
            }

        // Add the ghosted values
        ghosted_dst.compress(dealii::VectorOperation::add);
        dst = ghosted_dst;
      }
  }


  template <int dim, typename Number>
  void
  MatrixFree<dim, Number>::copy_constrained_values(
    const dealii::LinearAlgebra::distributed::Vector<Number,
                                                     dealii::MemorySpace::CUDA>
                                                                          &src,
    dealii::LinearAlgebra::distributed::Vector<Number,
                                               dealii::MemorySpace::CUDA> &dst)
    const
  {
    Assert(src.size() == dst.size(),
           dealii::ExcMessage(
             "src and dst vectors have different local size."));
    copy_constrained_dofs<Number>
      <<<constraint_grid_dim, constraint_block_dim>>>(constrained_dofs,
                                                      n_constrained_dofs,
                                                      src.locally_owned_size(),
                                                      src.get_values(),
                                                      dst.get_values());
    AssertCudaKernel();
  }



  template <int dim, typename Number>
  void
  MatrixFree<dim, Number>::set_constrained_values(
    const Number                                                           val,
    dealii::LinearAlgebra::distributed::Vector<Number,
                                               dealii::MemorySpace::CUDA> &dst)
    const
  {
    set_constrained_dofs<Number>
      <<<constraint_grid_dim, constraint_block_dim>>>(constrained_dofs,
                                                      n_constrained_dofs,
                                                      dst.locally_owned_size(),
                                                      val,
                                                      dst.get_values());
    AssertCudaKernel();
  }



} // namespace PSMF

  // #endif // CUDA_MATRIX_FREE_TEMPLATE_CUH