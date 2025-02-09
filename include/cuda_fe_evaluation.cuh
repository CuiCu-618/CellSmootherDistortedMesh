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

#include <deal.II/matrix_free/evaluation_flags.h>

#include "cuda_matrix_free.cuh"
#include "cuda_tensor_product_kernels.cuh"
#include <cuda/std/array>

#define UNIFORM_MESH

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
            dim == 2 ? threadIdx.x + n_points_1d * (threadIdx.y % n_points_1d) :
                       threadIdx.x + n_points_1d * (threadIdx.y % n_points_1d));
  }


  /**
   * For face integral, compute the dof/quad index for a given thread id,
   * dimension, and number of points in each space dimensions.
   */
  template <int dim, int n_points_1d>
  __device__ inline unsigned int
  compute_face_index(unsigned int face_number, unsigned int z = 0)
  {
    return (dim == 1 ?
              0 :
            dim == 2 ?
              (face_number == 0 ? threadIdx.y % n_points_1d : threadIdx.x) :
              (face_number == 0 ? threadIdx.y % n_points_1d + n_points_1d * z :
               face_number == 1 ? threadIdx.x + n_points_1d * z :
                                  threadIdx.x +
                                    n_points_1d * (threadIdx.y % n_points_1d)));

    // FIX: The following is the correct one with distorted mesh,
    // but somehow there is a bug in permutation when running with TensorCores.
    // face_number == 1 ? threadIdx.x * n_points_1d + z :
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
    static constexpr unsigned int n_dofs_z = dim == 3 ? fe_degree + 1 : 1;

    /**
     * An alias for scalar quantities.
     */
    using value_type = cuda::std::array<Number, n_dofs_z>;

    /**
     * An alias for vectorial quantities.
     */
    using gradient_type = cuda::std::array<value_type, dim>;

    /**
     * An alias for vectorial quantities.
     */
    using hessian_type = dealii::Tensor<2, dim, Number>;

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
     * Evaluate the function hessians of the FE function given at the DoF values
     * in the input vector at the quadrature points on the unit cell. The
     * function arguments specify which parts shall actually be computed. This
     * function needs to be called before the functions @p get_hessian() give
     * useful information.
     * @warning only the diagonal elements
     * @todo full hessian matrix
     */
    __device__ void
    evaluate_hessian();

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
    __device__ value_type
    get_trace_hessian() const;

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

    Number *inv_jac;

  private:
    dealii::types::global_dof_index *local_to_global;
    unsigned int                     n_cells;
    unsigned int                     padding_length;
    const unsigned int               mf_object_id;

    const bool use_coloring;

    Number *JxW;

    // Internal buffer
    Number *values;
    Number *gradients[dim];

    Number *shape_values;
    Number *shape_gradients;
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
    static constexpr unsigned int n_dofs_z = dim == 3 ? fe_degree + 1 : 1;

    /**
     * An alias for scalar quantities.
     */
    using value_type = cuda::std::array<Number, n_dofs_z>;

    /**
     * An alias for vectorial quantities.
     */
    using gradient_type = cuda::std::array<value_type, dim>;

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
    __device__ Number
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

    Number *shape_values;
    Number *shape_gradients;
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
    : n_cells(data -> n_cells)
    , padding_length(data->padding_length)
    , mf_object_id(data->id)
    , use_coloring(data->use_coloring)
    , values(shdata->values)
    , shape_values(shdata->shape_values)
    , shape_gradients(shdata->shape_gradients)
  {
#ifdef UNIFORM_MESH
    inv_jac = data->inv_jacobian;
    JxW     = data->JxW;
#else
    inv_jac = data->inv_jacobian + padding_length * cell_id;
    JxW     = data->JxW + padding_length * cell_id;
#endif

    local_to_global = data->local_to_global + cell_id;

    for (unsigned int i = 0; i < dim; ++i)
      gradients[i] = shdata->gradients[i];

#if MEMORY_TYPE == 0
    shape_values    = data->cell_face_shape_values;
    shape_gradients = data->cell_face_shape_gradients;
#elif MEMORY_TYPE == 2
    {
      const unsigned int idx = compute_index<dim, n_q_points_1d>();
#  if TENSORCORE == 2
      const unsigned int idx1 = idx ^ get_base<n_q_points_1d>(threadIdx.y, 0);
#  else
      const unsigned int idx1 = idx;
#  endif

      shape_values[idx1] = get_cell_shape_values<Number>(mf_object_id)[idx];
      shape_gradients[idx1] =
        get_cell_shape_gradients<Number>(mf_object_id)[idx];
    }
#endif
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

    const dealii::types::global_dof_index src_idx = local_to_global[0];

    for (unsigned int i = 0; i < n_dofs_z; ++i)
      {
        const unsigned int idx = compute_index<dim, n_q_points_1d>() +
                                 i * n_q_points_1d * n_q_points_1d;
#if TENSORCORE == 2
        const unsigned int idx1 = idx ^ get_base<n_q_points_1d>(threadIdx.y, i);
#else
        const unsigned int idx1 = idx;
#endif
        // Use the read-only data cache.
        values[idx1] = __ldg(&src[src_idx + idx]);
      }

    __syncthreads();
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

    const dealii::types::global_dof_index destination_idx = local_to_global[0];

    for (unsigned int i = 0; i < n_dofs_z; ++i)
      {
        const unsigned int idx = compute_index<dim, n_q_points_1d>() +
                                 i * n_q_points_1d * n_q_points_1d;
#if TENSORCORE == 2
        const unsigned int idx1 = idx ^ get_base<n_q_points_1d>(threadIdx.y, i);
#else
        const unsigned int idx1 = idx;
#endif
        if (use_coloring)
          dst[destination_idx + idx] += values[idx1];
        else
          atomicAdd(&dst[destination_idx + idx], values[idx1]);
      }
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
#if MEMORY_TYPE == 0
      evaluator_tensor_product(mf_object_id, shape_values, shape_gradients);
#elif MEMORY_TYPE == 1
      evaluator_tensor_product(mf_object_id,
                               get_cell_shape_values<Number>(mf_object_id),
                               get_cell_shape_gradients<Number>(mf_object_id));
#elif MEMORY_TYPE == 2
      evaluator_tensor_product(mf_object_id, shape_values, shape_gradients);
#endif
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
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    evaluate_hessian()
  {
    EvaluatorTensorProduct<EvaluatorVariant::evaluate_general,
                           dim,
                           fe_degree,
                           n_q_points_1d,
                           Number>
      evaluator_tensor_product(mf_object_id);
    evaluator_tensor_product.hessian_at_quad_pts(values, gradients);
    __syncthreads();
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
#if MEMORY_TYPE == 0
      evaluator_tensor_product(mf_object_id, shape_values, shape_gradients);
#elif MEMORY_TYPE == 1
      evaluator_tensor_product(mf_object_id,
                               get_cell_shape_values<Number>(mf_object_id),
                               get_cell_shape_gradients<Number>(mf_object_id));
#elif MEMORY_TYPE == 2
      evaluator_tensor_product(mf_object_id, shape_values, shape_gradients);
#endif
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
    value_type val;

    for (unsigned int i = 0; i < n_dofs_z; ++i)
      {
        const unsigned int q_point = i * n_q_points_1d * n_q_points_1d +
                                     compute_index<dim, n_q_points_1d>();
        val[i] = values[q_point];
      }
    return val;
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
    value_type val;

    for (unsigned int i = 0; i < n_dofs_z; ++i)
      {
        const unsigned int dof = i * n_q_points_1d * n_q_points_1d +
                                 compute_index<dim, n_q_points_1d>();
        val[i] = values[dof];
      }
    return val;
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
    for (unsigned int i = 0; i < n_dofs_z; ++i)
      {
        const unsigned int q_point = i * n_q_points_1d * n_q_points_1d +
                                     compute_index<dim, n_q_points_1d>();
#if TENSORCORE == 2
        const unsigned int q_point1 =
          q_point ^ get_base<n_q_points_1d>(threadIdx.y, i);
#else
        const unsigned int q_point1 = q_point;
#endif
        values[q_point] = val_in[i] * JxW[q_point1];
      }

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
    for (unsigned int i = 0; i < n_dofs_z; ++i)
      {
        const unsigned int dof = i * n_q_points_1d * n_q_points_1d +
                                 compute_index<dim, fe_degree + 1>();
        values[dof] = val_in[i];
      }
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

#ifdef UNIFORM_MESH
    const Number *inv_jacobian = &inv_jac[0];
#endif
    gradient_type grad;

    for (unsigned int i = 0; i < n_dofs_z; ++i)
      {
        const unsigned int q_point = i * n_q_points_1d * n_q_points_1d +
                                     compute_index<dim, n_q_points_1d>();
#ifndef UNIFORM_MESH
        const Number *inv_jacobian = &inv_jac[q_point];
#endif
        for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
          {
            Number tmp = 0.;
            for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
              tmp +=
                inv_jacobian[padding_length * n_cells * (dim * d_2 + d_1)] *
                gradients[d_2][q_point];
            grad[d_1][i] = tmp;
          }
      }

    return grad;
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
    get_trace_hessian() const
  {
    static_assert(n_components_ == 1, "This function only supports FE with one \
                  components");

    // TODO optimize if the mesh is uniform
    const unsigned int q_point = compute_index<dim, n_q_points_1d>();

    Number trace = 0.;
    for (unsigned int d = 0; d < dim; ++d)
      {
        trace += gradients[d][q_point];
      }

    return trace;
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
#ifdef UNIFORM_MESH
    const Number *inv_jacobian = &inv_jac[0];
#endif
    for (unsigned int i = 0; i < n_dofs_z; ++i)
      {
        const unsigned int q_point = (i * n_q_points_1d * n_q_points_1d +
                                      compute_index<dim, n_q_points_1d>());
#if TENSORCORE == 2
        const unsigned int q_point1 =
          q_point ^ get_base<n_q_points_1d>(threadIdx.y, i);
#else
        const unsigned int q_point1 = q_point;
#endif
#ifndef UNIFORM_MESH
        const Number *inv_jacobian = &inv_jac[q_point];
#endif
        for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
          {
            Number tmp = 0.;
            for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
              tmp +=
                inv_jacobian[n_cells * padding_length * (dim * d_1 + d_2)] *
                grad_in[d_2][i];
            gradients[d_1][q_point] = tmp * JxW[q_point1];
          }
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
    : n_faces(data -> n_faces)
    , n_cells(data->n_cells)
    , padding_length(data->padding_length)
    , face_padding_length(data->face_padding_length)
    , mf_object_id(data->id)
    , use_coloring(data->use_coloring)
    , is_interior_face(is_interior_face)
    , matrix_type(data->matrix_type)
    , shape_values(shdata->shape_values)
    , shape_gradients(shdata->shape_gradients)
  {
    // auto face_no = is_interior_face ? face_id : face_id + n_faces;
    auto face_no = face_id;

    cell_id = data->face2cell_id[face_no];

    local_to_global = data->local_to_global + cell_id;
    l_to_g_coarse   = data->l_to_g_coarse + cell_id;

#ifdef UNIFORM_MESH
    inv_jac = data->face_inv_jacobian;
    JxW     = data->face_JxW;
#else
    inv_jac = data->face_inv_jacobian + face_padding_length * face_no;
    JxW     = data->face_JxW + face_padding_length * face_no;
#endif

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

#if MEMORY_TYPE == 0
    shape_values    = data->cell_face_shape_values;
    shape_gradients = data->cell_face_shape_gradients;
#elif MEMORY_TYPE == 2
    {
      const unsigned int idx = compute_index<dim, n_q_points_1d>();
#  if TENSORCORE == 2
      const unsigned int idx1 = idx ^ get_base<n_q_points_1d>(threadIdx.y, 0);
#  else
      const unsigned int idx1 = idx;
#  endif

      shape_values[idx1] = get_cell_shape_values<Number>(mf_object_id)[idx];
      shape_values[idx1 + n_q_points_1d * n_q_points_1d] =
        get_face_shape_values<Number>(mf_object_id)[idx];
      shape_values[idx1 + n_q_points_1d * n_q_points_1d * 2] =
        get_face_shape_values<Number>(
          mf_object_id)[idx + n_q_points_1d * n_q_points_1d];

      shape_gradients[idx1] =
        get_cell_shape_gradients<Number>(mf_object_id)[idx];
      shape_gradients[idx1 + n_q_points_1d * n_q_points_1d] =
        get_face_shape_gradients<Number>(mf_object_id)[idx];
      shape_gradients[idx1 + n_q_points_1d * n_q_points_1d * 2] =
        get_face_shape_gradients<Number>(
          mf_object_id)[idx + n_q_points_1d * n_q_points_1d];
    }
#endif
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
    const dealii::types::global_dof_index src_idx = local_to_global[0];

    for (unsigned int i = 0; i < n_dofs_z; ++i)
      {
        const unsigned int idx = compute_index<dim, n_q_points_1d>() +
                                 i * n_q_points_1d * n_q_points_1d;
#if TENSORCORE == 2
        const unsigned int idx1 = idx ^ get_base<n_q_points_1d>(threadIdx.y, i);
#else
        const unsigned int idx1 = idx;
#endif
        // Use the read-only data cache.
        if (ignore_read)
          values[idx1] = 0;
        else
          values[idx1] = __ldg(&src[src_idx + idx]);
      }

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

    const dealii::types::global_dof_index destination_idx = l_to_g_coarse[0];

    for (unsigned int i = 0; i < n_dofs_z; ++i)
      {
        const unsigned int idx = compute_index<dim, n_q_points_1d>() +
                                 i * n_q_points_1d * n_q_points_1d;
#if TENSORCORE == 2
        const unsigned int idx1 = idx ^ get_base<n_q_points_1d>(threadIdx.y, i);
#else
        const unsigned int idx1 = idx;
#endif
        if (use_coloring)
          {
            if (!ignore_write)
              dst[destination_idx + idx] += values[idx1];
          }
        else
          {
            if (!ignore_write)
              atomicAdd(&dst[destination_idx + idx], values[idx1]);
          }
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
#if MEMORY_TYPE == 0
      evaluator_tensor_product(mf_object_id,
                               face_number,
                               subface_number,
                               shape_values,
                               shape_gradients);
#elif MEMORY_TYPE == 1
      evaluator_tensor_product(mf_object_id,
                               face_number,
                               subface_number,
                               get_cell_shape_values<Number>(mf_object_id),
                               get_cell_shape_gradients<Number>(mf_object_id));
#elif MEMORY_TYPE == 2
      evaluator_tensor_product(mf_object_id,
                               face_number,
                               subface_number,
                               shape_values,
                               shape_gradients);
#endif
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
#if MEMORY_TYPE == 0
      evaluator_tensor_product(mf_object_id,
                               face_number,
                               subface_number,
                               shape_values,
                               shape_gradients);
#elif MEMORY_TYPE == 1
      evaluator_tensor_product(mf_object_id,
                               face_number,
                               subface_number,
                               get_cell_shape_values<Number>(mf_object_id),
                               get_cell_shape_gradients<Number>(mf_object_id));
#elif MEMORY_TYPE == 2
      evaluator_tensor_product(mf_object_id,
                               face_number,
                               subface_number,
                               shape_values,
                               shape_gradients);
#endif
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
    value_type val;

    for (unsigned int i = 0; i < n_dofs_z; ++i)
      {
        const unsigned int q_point = i * n_q_points_1d * n_q_points_1d +
                                     compute_index<dim, n_q_points_1d>();
        val[i] = values[q_point];
      }
    return val;
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
    value_type val;

    for (unsigned int i = 0; i < n_dofs_z; ++i)
      {
        const unsigned int dof = i * n_q_points_1d * n_q_points_1d +
                                 compute_index<dim, fe_degree + 1>();
        val[i] = values[dof];
      }
    return val;
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
    for (unsigned int i = 0; i < n_dofs_z; ++i)
      {
        const unsigned int q_point = i * n_q_points_1d * n_q_points_1d +
                                     compute_index<dim, n_q_points_1d>();
        const unsigned int q_point_face =
          compute_face_index<dim, n_q_points_1d>(face_number / 2, i);
#if TENSORCORE == 2
        const unsigned int q_point_face1 =
          q_point_face ^
          get_face_base<n_q_points_1d>(face_number / 2, threadIdx.y, i);
#else
        const unsigned int q_point_face1 = q_point_face;
#endif

        values[q_point] = val_in[i] * JxW[q_point_face1];
      }

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
    for (unsigned int i = 0; i < n_dofs_z; ++i)
      {
        const unsigned int dof = i * n_q_points_1d * n_q_points_1d +
                                 compute_index<dim, fe_degree + 1>();
        values[dof] = val_in[i];
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
                                       Number>::gradient_type
  FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    get_gradient() const
  {
    static_assert(n_components_ == 1, "This function only supports FE with one \
                  components");

#ifdef UNIFORM_MESH
    const Number *inv_jacobian = &inv_jac[0];
#endif
    gradient_type grad;

    for (unsigned int i = 0; i < n_dofs_z; ++i)
      {
        const unsigned int q_point = i * n_q_points_1d * n_q_points_1d +
                                     compute_index<dim, n_q_points_1d>();
#ifndef UNIFORM_MESH
        const unsigned int q_point_face =
          compute_face_index<dim, n_q_points_1d>(face_number / 2, i);
#  if TENSORCORE == 2
        const unsigned int q_point_face1 =
          q_point_face ^
          get_face_base<n_q_points_1d>(face_number / 2, threadIdx.y, i);
#  else
        const unsigned int q_point_face1 = q_point_face;
#  endif
        const Number *inv_jacobian = &inv_jac[q_point_face1];
#endif

        for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
          {
            Number tmp = 0.;
            for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
              tmp += inv_jacobian[n_cells * face_padding_length *
                                  (dim * d_2 + d_1)] *
                     gradients[d_2][q_point];
            grad[d_1][i] = tmp;
          }
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
#ifdef UNIFORM_MESH
    const Number *inv_jacobian = &inv_jac[0];
#endif

    for (unsigned int i = 0; i < n_dofs_z; ++i)
      {
        const unsigned int q_point = i * n_q_points_1d * n_q_points_1d +
                                     compute_index<dim, n_q_points_1d>();
        const unsigned int q_point_face =
          compute_face_index<dim, n_q_points_1d>(face_number / 2, i);
#if TENSORCORE == 2
        const unsigned int q_point_face1 =
          q_point_face ^
          get_face_base<n_q_points_1d>(face_number / 2, threadIdx.y, i);
#else
        const unsigned int q_point_face1 = q_point_face;
#endif
#ifndef UNIFORM_MESH
        const Number *inv_jacobian = &inv_jac[q_point_face1];
#endif
        for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
          {
            Number tmp = 0.;
            for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
              tmp += inv_jacobian[n_cells * face_padding_length *
                                  (dim * d_1 + d_2)] *
                     grad_in[d_2][i];
            gradients[d_1][q_point] = tmp * JxW[q_point_face1];
          }
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

#ifdef UNIFORM_MESH
    const Number *normal_vector = &normal_vec[0];
#endif

    const Number coe = is_interior_face ? 1.0 : -1.0;

    gradient_type grad              = get_gradient();
    value_type    normal_derivative = {};

    for (unsigned int i = 0; i < n_dofs_z; ++i)
      {
#ifndef UNIFORM_MESH
        const unsigned int q_point_face =
          compute_face_index<dim, n_q_points_1d>(face_number / 2, i);
#  if TENSORCORE == 2
        const unsigned int q_point_face1 =
          q_point_face ^
          get_face_base<n_q_points_1d>(face_number / 2, threadIdx.y, i);
#  else
        const unsigned int q_point_face1 = q_point_face;
#  endif
        const Number *normal_vector = &normal_vec[q_point_face1];
#endif
        for (unsigned int d = 0; d < dim; ++d)
          normal_derivative[i] +=
            grad[d][i] * normal_vector[n_cells * face_padding_length * d] * coe;
      }

    return normal_derivative;
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
#ifdef UNIFORM_MESH
    const Number *normal_vector = &normal_vec[0];
    const Number *inv_jacobian  = &inv_jac[0];
#endif

    const Number coe = is_interior_face ? 1. : -1.;

    gradient_type normal_x_jacobian;

    for (unsigned int i = 0; i < n_dofs_z; ++i)
      {
        const unsigned int q_point = i * n_q_points_1d * n_q_points_1d +
                                     compute_index<dim, n_q_points_1d>();
        const unsigned int q_point_face =
          compute_face_index<dim, n_q_points_1d>(face_number / 2, i);
#if TENSORCORE == 2
        const unsigned int q_point_face1 =
          q_point_face ^
          get_face_base<n_q_points_1d>(face_number / 2, threadIdx.y, i);
#else
        const unsigned int q_point_face1 = q_point_face;
#endif
#ifndef UNIFORM_MESH
        const Number *normal_vector = &normal_vec[q_point_face1];
        const Number *inv_jacobian  = &inv_jac[q_point_face1];
#endif
        for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
          {
            Number tmp = 0.;
            for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
              tmp += inv_jacobian[n_cells * face_padding_length *
                                  (dim * d_1 + d_2)] *
                     normal_vector[n_cells * face_padding_length * d_2];
            normal_x_jacobian[d_1][i] = coe * tmp;
          }

        for (unsigned int d = 0; d < dim; ++d)
          gradients[d][q_point] =
            grad_in[i] * normal_x_jacobian[d][i] * JxW[q_point_face1];
      }

    __syncthreads();
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ Number
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
