/**
 * @file cuda_fe_evaluation.cuh
 * @brief FEEvaluation class.
 *
 * This class provides all the functions necessary to evaluate functions at
 * quadrature points and cell integrations. In functionality, this class is
 * similar to FEValues<dim>.
 *
 * @author Cu Cui
 * @date 2024-01-22
 * @version 0.1
 *
 * @remark
 * @note
 * @warning
 */


#ifndef CUDA_FE_EVALUATION_CUH
#define CUDA_FE_EVALUATION_CUH

#include <deal.II/base/config.h>

#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/matrix_free/cuda_hanging_nodes_internal.h>
#include <deal.II/matrix_free/evaluation_flags.h>

#include "cuda_matrix_free.cuh"
#include "cuda_tensor_product_kernels.cuh"


namespace PSMF
{
  /**
   * Compute the dof/quad index for a given thread id, dimension, and
   * number of points in each space dimensions.
   */
  template <int dim, int n_points_1d>
  __device__ inline unsigned int
  compute_index()
  {
    return (dim == 1 ? threadIdx.x % n_points_1d :
            dim == 2 ? threadIdx.x % n_points_1d + n_points_1d * threadIdx.y :
                       threadIdx.x % n_points_1d +
                         n_points_1d *
                           (threadIdx.y + n_points_1d * threadIdx.z));
  }


  /**
   * For face integral, compute the dof/quad index for a given thread id,
   * dimension, and number of points in each space dimensions.
   */
  template <int dim, int n_points_1d>
  __device__ inline unsigned int
  compute_face_index(unsigned int face_number)
  {
    return (
      dim == 1 ? 0 :
      dim == 2 ? (face_number == 0 ? threadIdx.y : threadIdx.x % n_points_1d) :
                 (face_number == 0 ?
                    threadIdx.y + n_points_1d * threadIdx.z :
                  face_number == 1 ?
                    ((threadIdx.x % n_points_1d) * n_points_1d) + threadIdx.z :
                    threadIdx.x % n_points_1d + n_points_1d * threadIdx.y));
  }

  /**
   * This class provides all the functions necessary to evaluate functions at
   * quadrature points and cell integrations. In functionality, this class is
   * similar to FEValues<dim>.
   *
   * This class has five template arguments:
   *
   * @tparam dim Dimension in which this class is to be used
   *
   * @tparam fe_degree Degree of the tensor prodict finite element with fe_degree+1
   * degrees of freedom per coordinate direction
   *
   * @tparam n_q_points_1d Number of points in the quadrature formular in 1D,
   * defaults to fe_degree+1
   *
   * @tparam n_components Number of vector components when solving a system of
   * PDEs. If the same operation is applied to several components of a PDE (e.g.
   * a vector Laplace equation), they can be applied simultaneously with one
   * call (and often more efficiently). Defaults to 1
   *
   * @tparam Number Number format, @p double or @p float. Defaults to @p
   * double.
   *
   * @ingroup MatrixFree
   */
  template <int dim,
            int fe_degree,
            int n_q_points_1d = fe_degree + 1,
            int n_components_ = 1,
            typename Number   = double>
  class FEEvaluation
  {
  public:
    /**
     * An alias for scalar quantities.
     */
    using value_type = Number;

    /**
     * An alias for vectorial quantities.
     */
    using gradient_type = dealii::Tensor<1, dim, Number>;

    /**
     * An alias to kernel specific information.
     */
    using data_type = typename MatrixFree<dim, Number>::Data;

    /**
     * Dimension.
     */
    static constexpr unsigned int dimension = dim;

    /**
     * Number of components.
     */
    static constexpr unsigned int n_components = n_components_;

    /**
     * Number of quadrature points per cell.
     */
    static constexpr unsigned int n_q_points =
      dealii::Utilities::pow(n_q_points_1d, dim);

    /**
     * Number of tensor degrees of freedoms per cell.
     */
    static constexpr unsigned int tensor_dofs_per_cell =
      dealii::Utilities::pow(fe_degree + 1, dim);

    /**
     * Constructor.
     */
    __device__
    FEEvaluation(const unsigned int       cell_id,
                 const data_type         *data,
                 SharedData<dim, Number> *shdata);

    /**
     * For the vector @p src, read out the values on the degrees of freedom of
     * the current cell, and store them internally. Similar functionality as
     * the function DoFAccessor::get_interpolated_dof_values when no
     * constraints are present, but it also includes constraints from hanging
     * nodes, so once can see it as a similar function to
     * AffineConstraints::read_dof_valuess as well.
     */
    __device__ void
    read_dof_values(const Number *src);

    /**
     * Take the value stored internally on dof values of the current cell and
     * sum them into the vector @p dst. The function also applies constraints
     * during the write operation. The functionality is hence similar to the
     * function AffineConstraints::distribute_local_to_global.
     */
    __device__ void
    distribute_local_to_global(Number *dst) const;

    /**
     * Evaluate the function values and the gradients of the FE function given
     * at the DoF values in the input vector at the quadrature points on the
     * unit cell. The function arguments specify which parts shall actually be
     * computed. This function needs to be called before the functions
     * @p get_value() or @p get_gradient() give useful information.
     */
    __device__ void
    evaluate(const bool evaluate_val, const bool evaluate_grad);

    /**
     * This function takes the values and/or gradients that are stored on
     * quadrature points, tests them by all the basis functions/gradients on
     * the cell and performs the cell integration. The two function arguments
     * @p integrate_val and @p integrate_grad are used to enable/disable some
     * of the values or the gradients.
     */
    __device__ void
    integrate(const bool integrate_val, const bool integrate_grad);

    /**
     * Same as above, except that the quadrature point is computed from thread
     * id.
     */
    __device__ value_type
    get_value() const;

    /**
     * Same as above, except that the local dof index is computed from the
     * thread id.
     */
    __device__ value_type
    get_dof_value() const;

    /**
     * Same as above, except that the quadrature point is computed from the
     * thread id.
     */
    __device__ void
    submit_value(const value_type &val_in);

    /**
     * Same as above, except that the local dof index is computed from the
     * thread id.
     */
    __device__ void
    submit_dof_value(const value_type &val_in);

    /**
     * Same as above, except that the quadrature point is computed from the
     * thread id.
     */
    __device__ gradient_type
    get_gradient() const;

    /**
     * Same as above, except that the quadrature point is computed from the
     * thread id.
     */
    __device__ void
    submit_gradient(const gradient_type &grad_in);

    // clang-format off
    /**
     * Same as above, except that the functor @p func only takes a single input
     * argument (fe_eval) and computes the quadrature point from the thread id.
     *
     * @p func needs to define
     * \code
     * __device__ void operator()(
     *   CUDAWrappers::FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number> *fe_eval) const;
     * \endcode
     */
    // clang-format on
    template <typename Functor>
    __device__ void
    apply_for_each_quad_point(const Functor &func);

  private:
    dealii::types::global_dof_index *local_to_global;
    unsigned int                     n_cells;
    unsigned int                     padding_length;
    const unsigned int               mf_object_id;

    const dealii::internal::MatrixFreeFunctions::ConstraintKinds
      constraint_mask;

    const bool use_coloring;

    Number *inv_jac;
    Number *JxW;

    // Internal buffer
    Number *values;
    Number *gradients[dim];
  };



  /**
   * This class provides all the functions necessary to evaluate functions at
   * quadrature points and cell/face integrations. In functionality, this class
   * is similar to FEFaceValues<dim>.
   *
   * This class has five template arguments:
   *
   * @tparam dim Dimension in which this class is to be used
   *
   * @tparam fe_degree Degree of the tensor prodict finite element with fe_degree+1
   * degrees of freedom per coordinate direction
   *
   * @tparam n_q_points_1d Number of points in the quadrature formular in 1D,
   * defaults to fe_degree+1
   *
   * @tparam n_components Number of vector components when solving a system of
   * PDEs. If the same operation is applied to several components of a PDE (e.g.
   * a vector Laplace equation), they can be applied simultaneously with one
   * call (and often more efficiently). Defaults to 1
   *
   * @tparam Number Number format, @p double or @p float. Defaults to @p
   * double.
   *
   * @ingroup MatrixFree
   */
  template <int dim,
            int fe_degree,
            int n_q_points_1d = fe_degree + 1,
            int n_components_ = 1,
            typename Number   = double>
  class FEFaceEvaluation
  {
  public:
    /**
     * An alias for scalar quantities.
     */
    using value_type = Number;

    /**
     * An alias for vectorial quantities.
     */
    using gradient_type = dealii::Tensor<1, dim, Number>;

    /**
     * An alias to kernel specific information.
     */
    using data_type = typename MatrixFree<dim, Number>::Data;

    /**
     * Dimension.
     */
    static constexpr unsigned int dimension = dim;

    /**
     * Number of components.
     */
    static constexpr unsigned int n_components = n_components_;

    /**
     * Number of quadrature points per cell.
     */
    static constexpr unsigned int n_q_points =
      dealii::Utilities::pow(n_q_points_1d, dim - 1);

    /**
     * Number of tensor degrees of freedoms per cell.
     */
    static constexpr unsigned int tensor_dofs_per_cell =
      dealii::Utilities::pow(fe_degree + 1, dim);

    /**
     * Constructor.
     */
    __device__
    FEFaceEvaluation(const unsigned int       face_id,
                     const data_type         *data,
                     SharedData<dim, Number> *shdata,
                     const bool               is_interior_face = true);

    /**
     * For the vector @p src, read out the values on the degrees of freedom of
     * the current cell, and store them internally. Similar functionality as
     * the function DoFAccessor::get_interpolated_dof_values when no
     * constraints are present, but it also includes constraints from hanging
     * nodes, so once can see it as a similar function to
     * AffineConstraints::read_dof_valuess as well.
     */
    __device__ void
    read_dof_values(const Number *src);

    /**
     * Take the value stored internally on dof values of the current cell and
     * sum them into the vector @p dst. The function also applies constraints
     * during the write operation. The functionality is hence similar to the
     * function AffineConstraints::distribute_local_to_global.
     */
    __device__ void
    distribute_local_to_global(Number *dst) const;

    /**
     * Evaluates the function values, the gradients, and the Laplacians of the
     * FE function given at the DoF values stored in the internal data field
     * dof_values (that is usually filled by the read_dof_values() method) at
     * the quadrature points on the unit cell. The function arguments specify
     * which parts shall actually be computed. Needs to be called before the
     * functions get_value(), get_gradient() or get_normal_derivative() give
     * useful information (unless these values have been set manually by
     * accessing the internal data pointers).
     */
    __device__ void
    evaluate(const bool evaluate_val, const bool evaluate_grad);

    /**
     * This function takes the values and/or gradients that are stored on
     * quadrature points, tests them by all the basis functions/gradients on the
     * cell and performs the cell integration. The two function arguments
     * integrate_val and integrate_grad are used to enable/disable some of
     * values or gradients. The result is written into the internal data field
     * dof_values (that is usually written into the result vector by the
     * distribute_local_to_global() or set_dof_values() methods).
     */
    __device__ void
    integrate(const bool integrate_val, const bool integrate_grad);

    /**
     * Same as above, except that the quadrature point is computed from thread
     * id.
     */
    __device__ value_type
    get_value() const;

    /**
     * Same as above, except that the local dof index is computed from the
     * thread id.
     */
    __device__ value_type
    get_dof_value() const;

    /**
     * Same as above, except that the quadrature point is computed from the
     * thread id.
     */
    __device__ void
    submit_value(const value_type &val_in);

    /**
     * Same as above, except that the local dof index is computed from the
     * thread id.
     */
    __device__ void
    submit_dof_value(const value_type &val_in);

    /**
     * Same as above, except that the quadrature point is computed from the
     * thread id.
     */
    __device__ gradient_type
    get_gradient() const;

    /**
     * Same as above, except that the quadrature point is computed from the
     * thread id.
     */
    __device__ void
    submit_gradient(const gradient_type &grad_in);

    /**
     * Same as above, except that the quadrature point is computed from the
     * thread id.
     */
    __device__ value_type
    get_normal_derivative() const;

    /**
     * Same as above, except that the quadrature point is computed from the
     * thread id.
     */
    __device__ void
    submit_normal_derivative(const value_type &grad_in);

    /**
     * length h_i normal to the face. For a general non-Cartesian mesh, this
     * length must be computed by the product of the inverse Jacobian times the
     * normal vector in real coordinates.
     */
    __device__ value_type
    inverse_length_normal_to_face();

    // clang-format off
    /**
     * Same as above, except that the functor @p func only takes a single input
     * argument (fe_eval) and computes the quadrature point from the thread id.
     *
     * @p func needs to define
     * \code
     * __device__ void operator()(
     *   CUDAWrappers::FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number> *fe_eval) const;
     * \endcode
     */
    // clang-format on
    template <typename Functor>
    __device__ void
    apply_for_each_quad_point(const Functor &func);

    Number *JxW;
    Number *inv_jac;
    Number *normal_vec;

  private:
    dealii::types::global_dof_index *local_to_global;
    dealii::types::global_dof_index *l_to_g_coarse;
    dealii::types::global_dof_index *face_to_cell;
    unsigned int                     cell_id;
    unsigned int                     n_faces;
    unsigned int                     n_cells;
    unsigned int                     padding_length;
    unsigned int                     face_padding_length;
    unsigned int                     face_number;
    int                              subface_number;
    int                              face_orientation;
    bool                             ignore_read;
    bool                             ignore_write;


    const unsigned int mf_object_id;
    const bool         use_coloring;
    const bool         is_interior_face;
    const MatrixType   matrix_type;

    // Internal buffer
    Number *values;
    Number *gradients[dim];
  };



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    FEEvaluation(const unsigned int       cell_id,
                 const data_type         *data,
                 SharedData<dim, Number> *shdata)
    : n_cells(data->n_cells)
    , padding_length(data->padding_length)
    , mf_object_id(data->id)
    , constraint_mask(data->constraint_mask[cell_id])
    , use_coloring(data->use_coloring)
    , values(shdata->values)
  {
    local_to_global = data->local_to_global + padding_length * cell_id;
    inv_jac         = data->inv_jacobian + padding_length * cell_id;
    JxW             = data->JxW + padding_length * cell_id;

    for (unsigned int i = 0; i < dim; ++i)
      gradients[i] = shdata->gradients[i];
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    read_dof_values(const Number *src)
  {
    static_assert(n_components_ == 1, "This function only supports FE with one \
                  components");
    const unsigned int idx = compute_index<dim, n_q_points_1d>();

    const dealii::types::global_dof_index src_idx = local_to_global[idx];
    // Use the read-only data cache.
    values[idx] = __ldg(&src[src_idx]);

    __syncthreads();

    dealii::CUDAWrappers::internal::
      resolve_hanging_nodes<dim, fe_degree, false>(constraint_mask, values);
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    distribute_local_to_global(Number *dst) const
  {
    static_assert(n_components_ == 1, "This function only supports FE with one \
                  components");
    dealii::CUDAWrappers::internal::resolve_hanging_nodes<dim, fe_degree, true>(
      constraint_mask, values);

    const unsigned int idx = compute_index<dim, n_q_points_1d>();

    const dealii::types::global_dof_index destination_idx =
      local_to_global[idx];

    if (use_coloring)
      dst[destination_idx] += values[idx];
    else
      atomicAdd(&dst[destination_idx], values[idx]);
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::evaluate(
    const bool evaluate_val,
    const bool evaluate_grad)
  {
    // First evaluate the gradients because it requires values that will be
    // changed if evaluate_val is true
    EvaluatorTensorProduct<EvaluatorVariant::evaluate_general,
                           dim,
                           fe_degree,
                           n_q_points_1d,
                           Number>
      evaluator_tensor_product(mf_object_id);
    if (evaluate_val == true && evaluate_grad == true)
      {
        evaluator_tensor_product.value_and_gradient_at_quad_pts(values,
                                                                gradients);
        __syncthreads();
      }
    else if (evaluate_grad == true)
      {
        evaluator_tensor_product.gradient_at_quad_pts(values, gradients);
        __syncthreads();
      }
    else if (evaluate_val == true)
      {
        evaluator_tensor_product.value_at_quad_pts(values);
        __syncthreads();
      }
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::integrate(
    const bool integrate_val,
    const bool integrate_grad)
  {
    EvaluatorTensorProduct<EvaluatorVariant::evaluate_general,
                           dim,
                           fe_degree,
                           n_q_points_1d,
                           Number>
      evaluator_tensor_product(mf_object_id);
    if (integrate_val == true && integrate_grad == true)
      {
        evaluator_tensor_product.integrate_value_and_gradient(values,
                                                              gradients);
        __syncthreads();
      }
    else if (integrate_val == true)
      {
        evaluator_tensor_product.integrate_value(values);
        __syncthreads();
      }
    else if (integrate_grad == true)
      {
        evaluator_tensor_product.integrate_gradient<false>(values, gradients);
        __syncthreads();
      }
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ typename FEEvaluation<dim,
                                   fe_degree,
                                   n_q_points_1d,
                                   n_components_,
                                   Number>::value_type
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    get_value() const
  {
    const unsigned int q_point = compute_index<dim, n_q_points_1d>();
    return values[q_point];
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ typename FEEvaluation<dim,
                                   fe_degree,
                                   n_q_points_1d,
                                   n_components_,
                                   Number>::value_type
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    get_dof_value() const
  {
    const unsigned int dof = compute_index<dim, fe_degree + 1>();
    return values[dof];
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    submit_value(const value_type &val_in)
  {
    const unsigned int q_point = compute_index<dim, n_q_points_1d>();
    values[q_point]            = val_in * JxW[q_point];
    __syncthreads();
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    submit_dof_value(const value_type &val_in)
  {
    const unsigned int dof = compute_index<dim, fe_degree + 1>();
    values[dof]            = val_in;
    __syncthreads();
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ typename FEEvaluation<dim,
                                   fe_degree,
                                   n_q_points_1d,
                                   n_components_,
                                   Number>::gradient_type
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    get_gradient() const
  {
    static_assert(n_components_ == 1, "This function only supports FE with one \
                  components");

    // TODO optimize if the mesh is uniform
    const unsigned int q_point      = compute_index<dim, n_q_points_1d>();
    const Number      *inv_jacobian = &inv_jac[q_point];
    gradient_type      grad;
    for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
      {
        Number tmp = 0.;
        for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
          tmp += inv_jacobian[padding_length * n_cells * (dim * d_2 + d_1)] *
                 gradients[d_2][q_point];
        grad[d_1] = tmp;
      }

    return grad;
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    submit_gradient(const gradient_type &grad_in)
  {
    // TODO optimize if the mesh is uniform
    const unsigned int q_point      = compute_index<dim, n_q_points_1d>();
    const Number      *inv_jacobian = &inv_jac[q_point];
    for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
      {
        Number tmp = 0.;
        for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
          tmp += inv_jacobian[n_cells * padding_length * (dim * d_1 + d_2)] *
                 grad_in[d_2];
        gradients[d_1][q_point] = tmp * JxW[q_point];
      }
    __syncthreads();
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  template <typename Functor>
  __device__ void
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    apply_for_each_quad_point(const Functor &func)
  {
    func(this);

    __syncthreads();
  }


  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__
  FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    FEFaceEvaluation(const unsigned int       face_id,
                     const data_type         *data,
                     SharedData<dim, Number> *shdata,
                     const bool               is_interior_face)
    : n_faces(data->n_faces)
    , n_cells(data->n_cells)
    , padding_length(data->padding_length)
    , face_padding_length(data->face_padding_length)
    , mf_object_id(data->id)
    , use_coloring(data->use_coloring)
    , is_interior_face(is_interior_face)
    , matrix_type(data->matrix_type)
  {
    auto face_no = is_interior_face ? face_id : face_id + n_faces;

    cell_id = data->face2cell_id[face_no];

    local_to_global = data->local_to_global + padding_length * cell_id;
    l_to_g_coarse   = data->l_to_g_coarse + padding_length * cell_id;

    inv_jac        = data->face_inv_jacobian + face_padding_length * face_no;
    JxW            = data->face_JxW + face_padding_length * face_no;
    normal_vec     = data->normal_vector + face_padding_length * face_no;
    face_number    = data->face_number[face_no];
    subface_number = data->subface_number[face_no];

    unsigned int shift = is_interior_face ? 0 : tensor_dofs_per_cell;

    ignore_read  = false;
    ignore_write = false;

    if (matrix_type == MatrixType::level_matrix)
      {
        ignore_read  = !is_interior_face && subface_number != -1;
        ignore_write = ignore_read;
      }
    else if (matrix_type == MatrixType::edge_down_matrix)
      {
        ignore_read  = !is_interior_face || subface_number == -1;
        ignore_write = is_interior_face || subface_number == -1;
      }
    else if (matrix_type == MatrixType::edge_up_matrix)
      {
        ignore_read  = is_interior_face || subface_number == -1;
        ignore_write = !is_interior_face || subface_number == -1;
      }

    values = &shdata->values[shift];

    for (unsigned int i = 0; i < dim; ++i)
      gradients[i] = &shdata->gradients[i][shift];
  }

  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    read_dof_values(const Number *src)
  {
    static_assert(n_components_ == 1, "This function only supports FE with one \
                  components");
    const unsigned int idx = compute_index<dim, n_q_points_1d>();

    const dealii::types::global_dof_index src_idx = local_to_global[idx];
    // Use the read-only data cache.
    if (ignore_read)
      values[idx] = 0;
    else
      values[idx] = __ldg(&src[src_idx]);

    __syncthreads();
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    distribute_local_to_global(Number *dst) const
  {
    static_assert(n_components_ == 1, "This function only supports FE with one \
                  components");

    const unsigned int idx = compute_index<dim, n_q_points_1d>();

    const dealii::types::global_dof_index destination_idx = l_to_g_coarse[idx];

    if (use_coloring)
      {
        if (!ignore_write)
          dst[destination_idx] += values[idx];
      }
    else
      {
        if (!ignore_write)
          atomicAdd(&dst[destination_idx], values[idx]);
      }
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    evaluate(const bool evaluate_val, const bool evaluate_grad)
  {
    // First evaluate the gradients because it requires values that will be
    // changed if evaluate_val is true
    EvaluatorTensorProduct<EvaluatorVariant::evaluate_face,
                           dim,
                           fe_degree,
                           n_q_points_1d,
                           Number>
      evaluator_tensor_product(mf_object_id, face_number, subface_number);
    if (evaluate_val == true && evaluate_grad == true)
      {
        // todo:
        // evaluator_tensor_product.value_and_gradient_at_quad_pts(values,
        //                                                         gradients);

        evaluator_tensor_product.gradient_at_quad_pts(values, gradients);
        __syncthreads();

        evaluator_tensor_product.value_at_quad_pts(values);

        __syncthreads();
      }
    else if (evaluate_grad == true)
      {
        evaluator_tensor_product.gradient_at_quad_pts(values, gradients);
        __syncthreads();
      }
    else if (evaluate_val == true)
      {
        evaluator_tensor_product.value_at_quad_pts(values);
        __syncthreads();
      }
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    integrate(const bool integrate_val, const bool integrate_grad)
  {
    // First evaluate the gradients because it requires values that will be
    // changed if evaluate_val is true
    EvaluatorTensorProduct<EvaluatorVariant::evaluate_face,
                           dim,
                           fe_degree,
                           n_q_points_1d,
                           Number>
      evaluator_tensor_product(mf_object_id, face_number, subface_number);
    if (integrate_val == true && integrate_grad == true)
      {
        // todo
        // evaluator_tensor_product.integrate_value_and_gradient(values,
        //                                                       gradients);

        evaluator_tensor_product.integrate_value(values);
        __syncthreads();

        evaluator_tensor_product.integrate_gradient<true>(values, gradients);
        __syncthreads();
      }
    else if (integrate_val == true)
      {
        evaluator_tensor_product.integrate_value(values);
        __syncthreads();
      }
    else if (integrate_grad == true)
      {
        evaluator_tensor_product.integrate_gradient<false>(values, gradients);
        __syncthreads();
      }
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ typename FEFaceEvaluation<dim,
                                       fe_degree,
                                       n_q_points_1d,
                                       n_components_,
                                       Number>::value_type
  FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    get_value() const
  {
    const unsigned int q_point = compute_index<dim, n_q_points_1d>();
    return values[q_point];
  }


  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ typename FEFaceEvaluation<dim,
                                       fe_degree,
                                       n_q_points_1d,
                                       n_components_,
                                       Number>::value_type
  FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    get_dof_value() const
  {
    const unsigned int dof = compute_index<dim, fe_degree + 1>();
    return values[dof];
  }


  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    submit_value(const value_type &val_in)
  {
    const unsigned int q_point = compute_index<dim, n_q_points_1d>();
    const unsigned int q_point_face =
      compute_face_index<dim, n_q_points_1d>(face_number / 2);

    values[q_point] = val_in * JxW[q_point_face];

    __syncthreads();
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    submit_dof_value(const value_type &val_in)
  {
    const unsigned int dof = compute_index<dim, fe_degree + 1>();
    values[dof]            = val_in;

    __syncthreads();
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ typename FEFaceEvaluation<dim,
                                       fe_degree,
                                       n_q_points_1d,
                                       n_components_,
                                       Number>::gradient_type
  FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    get_gradient() const
  {
    static_assert(n_components_ == 1, "This function only supports FE with one \
                  components");

    // TODO optimize if the mesh is uniform
    const unsigned int q_point = compute_index<dim, n_q_points_1d>();
    const unsigned int q_point_face =
      compute_face_index<dim, n_q_points_1d>(face_number / 2);
    const Number *inv_jacobian = &inv_jac[q_point_face];
    gradient_type grad;
    for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
      {
        Number tmp = 0.;
        for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
          tmp +=
            inv_jacobian[n_cells * face_padding_length * (dim * d_2 + d_1)] *
            gradients[d_2][q_point];
        grad[d_1] = tmp;
      }

    return grad;
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    submit_gradient(const gradient_type &grad_in)
  {
    // TODO optimize if the mesh is uniform
    const unsigned int q_point = compute_index<dim, n_q_points_1d>();
    const unsigned int q_point_face =
      compute_face_index<dim, n_q_points_1d>(face_number / 2);
    const Number *inv_jacobian = &inv_jac[q_point_face];
    for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
      {
        Number tmp = 0.;
        for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
          tmp +=
            inv_jacobian[n_cells * face_padding_length * (dim * d_1 + d_2)] *
            grad_in[d_2];
        gradients[d_1][q_point] = tmp * JxW[q_point_face];
      }

    __syncthreads();
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ typename FEFaceEvaluation<dim,
                                       fe_degree,
                                       n_q_points_1d,
                                       n_components_,
                                       Number>::value_type
  FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    get_normal_derivative() const
  {
    static_assert(n_components_ == 1, "This function only supports FE with one \
                  components");

    // TODO optimize if the mesh is uniform
    const unsigned int q_point_face =
      compute_face_index<dim, n_q_points_1d>(face_number / 2);
    const Number *normal_vector = &normal_vec[q_point_face];

    gradient_type grad              = get_gradient();
    value_type    normal_derivative = 0.;

    for (unsigned int d = 0; d < dim; ++d)
      normal_derivative +=
        grad[d] * normal_vector[n_cells * face_padding_length * d];

    return is_interior_face ? normal_derivative : -normal_derivative;
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    submit_normal_derivative(const value_type &grad_in)
  {
    // TODO optimize if the mesh is uniform
    const unsigned int q_point = compute_index<dim, n_q_points_1d>();
    const unsigned int q_point_face =
      compute_face_index<dim, n_q_points_1d>(face_number / 2);
    const Number *normal_vector = &normal_vec[q_point_face];
    const Number *inv_jacobian  = &inv_jac[q_point_face];

    const Number coe = is_interior_face ? 1. : -1.;

    gradient_type normal_x_jacobian;
    for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
      {
        Number tmp = 0.;
        for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
          tmp +=
            inv_jacobian[n_cells * face_padding_length * (dim * d_1 + d_2)] *
            normal_vector[n_cells * face_padding_length * d_2];
        normal_x_jacobian[d_1] = coe * tmp;
      }

    for (unsigned int d = 0; d < dim; ++d)
      gradients[d][q_point] =
        grad_in * normal_x_jacobian[d] * JxW[q_point_face];

    __syncthreads();
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ typename FEFaceEvaluation<dim,
                                       fe_degree,
                                       n_q_points_1d,
                                       n_components_,
                                       Number>::value_type
  FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    inverse_length_normal_to_face()
  {
    Number tmp = 0.;
    for (unsigned int d = 0; d < dim; ++d)
      tmp +=
        inv_jac[n_cells * face_padding_length * (dim * (face_number / 2) + d)] *
        normal_vec[n_cells * face_padding_length * d];

    return tmp;
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  template <typename Functor>
  __device__ void
  FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    apply_for_each_quad_point(const Functor &func)
  {
    func(this);

    __syncthreads();
  }

} // namespace PSMF

#endif // CUDA_FE_EVALUATION_CUH