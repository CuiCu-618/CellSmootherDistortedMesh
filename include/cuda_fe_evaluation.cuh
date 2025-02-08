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
#include <cuda/std/array>

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
   * Compute the dof/quad index for a given thread id, dimension, and
   * number of points in each space dimensions.
   */
  template <int dim, int n_points_1d>
  __device__ inline unsigned int
  compute_active_index()
  {
    constexpr unsigned int n_q_points_1d_1 = n_points_1d - 1;

    return (dim == 1 ? threadIdx.x % n_q_points_1d_1 :
            dim == 2 ? threadIdx.x % n_q_points_1d_1 +
                         n_q_points_1d_1 * threadIdx.y :
                       threadIdx.x % n_q_points_1d_1 +
                         n_q_points_1d_1 *
                           (threadIdx.y + n_q_points_1d_1 * threadIdx.z));
  }

  template <int dim, int n_points_1d>
  __device__ inline bool
  is_active()
  {
    constexpr unsigned int n_q_points_1d_1 = n_points_1d - 1;

    return (threadIdx.x % n_points_1d < n_q_points_1d_1 &&
            threadIdx.y < n_q_points_1d_1 && threadIdx.z < n_q_points_1d_1);
  }

  /**
   * Compute real dof index for active thread (based on function is_active()).
   */
  template <int dim, int n_points_1d>
  __device__ inline unsigned int
  compute_active_index(const unsigned int c)
  {
    constexpr unsigned int n_q_points_1d_1 = n_points_1d - 1;

    return (
      c == 0 ? threadIdx.x % n_points_1d +
                 n_points_1d * (threadIdx.y + n_q_points_1d_1 * threadIdx.z) :
      c == 1 ? threadIdx.x % n_q_points_1d_1 +
                 n_q_points_1d_1 * (threadIdx.y + n_points_1d * threadIdx.z) :
               threadIdx.x % n_q_points_1d_1 +
                 n_q_points_1d_1 *
                   (threadIdx.y + n_q_points_1d_1 * threadIdx.z));
  }

  template <int dim, int n_points_1d>
  __device__ inline bool
  is_active(const unsigned int c)
  {
    constexpr unsigned int n_q_points_1d_1 = n_points_1d - 1;

    return (c == 0 ?
              threadIdx.y < n_q_points_1d_1 && threadIdx.z < n_q_points_1d_1 :
            c == 1 ? threadIdx.x % n_points_1d < n_q_points_1d_1 &&
                       threadIdx.z < n_q_points_1d_1 :
                     threadIdx.x % n_points_1d < n_q_points_1d_1 &&
                       threadIdx.y < n_q_points_1d_1);
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
    using value_type = cuda::std::array<Number, n_components_>;

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
      dealii::Utilities::pow(n_q_points_1d, dim);

    /**
     * Number of tensor degrees of freedoms per cell.
     */
    static constexpr unsigned int tensor_dofs_per_cell =
      dealii::Utilities::pow(fe_degree + 1, dim);

    static constexpr unsigned int rt_tensor_dofs_per_cell =
      dealii::Utilities::pow(fe_degree + 2, dim);

    static constexpr unsigned int rt_tensor_dofs_per_component =
      dealii::Utilities::pow(fe_degree + 1, dim - 1) * (fe_degree + 2);

    /**
     * Constructor.
     */
    __device__
    FEEvaluation(const unsigned int       cell_id,
                 const data_type         &data,
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
     * Same as above, except that the quadrature point is computed from thread
     * id.
     */
    __device__ Number
    get_divergence() const;

    /**
     * Same as above, except that the quadrature point is computed from the
     * thread id.
     */
    __device__ void
    submit_divergence(const Number &div_in);

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

    dealii::internal::MatrixFreeFunctions::ConstraintKinds *constraint_mask;

    const dealii::types::global_dof_index *hanging_nodes_constraint;
    const int                             *hanging_nodes_constraint_indicator;
    const Number                          *hanging_nodes_constraint_weights;
    unsigned int                          *constraint_range;

    const bool use_coloring;
    const bool is_primitive;

    Number *jac;
    Number *inv_jac;
    Number *JxW;
    Number *weights;
    Number *inv_det;

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
    using value_type = cuda::std::array<Number, n_components_>;

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

    static constexpr unsigned int rt_tensor_dofs_per_cell =
      dealii::Utilities::pow(fe_degree + 2, dim);

    static constexpr unsigned int rt_tensor_dofs_per_component =
      dealii::Utilities::pow(fe_degree + 1, dim - 1) * (fe_degree + 2);

    /**
     * Constructor.
     */
    __device__
    FEFaceEvaluation(const unsigned int       face_id,
                     const data_type         &data,
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
    Number *jac;
    Number *inv_jac;
    Number *weights;
    Number *inv_det;
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

    const dealii::types::global_dof_index *hanging_nodes_constraint;
    const int                             *hanging_nodes_constraint_indicator;
    const Number                          *hanging_nodes_constraint_weights;
    unsigned int                          *constraint_range;

    unsigned int                          *constraint_coarse_range;
    const dealii::types::global_dof_index *hanging_nodes_constraint_coarse;

    const unsigned int mf_object_id;
    const bool         use_coloring;
    const bool         is_primitive;
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
                 const data_type         &data,
                 SharedData<dim, Number> *shdata)
    : n_cells(data.n_cells)
    , padding_length(data.padding_length)
    , mf_object_id(data.id)
    , hanging_nodes_constraint(data.hanging_nodes_constraint)
    , hanging_nodes_constraint_indicator(
        data.hanging_nodes_constraint_indicator)
    , hanging_nodes_constraint_weights(data.hanging_nodes_constraint_weights)
    , use_coloring(data.use_coloring)
    , is_primitive(data.is_primitive)
    , values(shdata->values)
  {
    constraint_mask  = data.constraint_mask + cell_id * n_components;
    constraint_range = data.constraint_range + cell_id * 2;

    local_to_global =
      data.local_to_global + padding_length * cell_id * n_components;
    jac     = data.jacobian + padding_length * cell_id;
    inv_jac = data.inv_jacobian + padding_length * cell_id;
    JxW     = data.JxW + padding_length * cell_id;
    weights = data.q_weights;
    inv_det = data.inv_det + padding_length * cell_id;

    for (unsigned int i = 0; i < dim; ++i)
      gradients[i] = shdata->gradients[i];


    const unsigned int idx    = compute_index<dim, n_q_points_1d>();
    const unsigned int stride = rt_tensor_dofs_per_cell;
    for (unsigned int c = 0; c < n_components_; ++c)
      {
        values[idx + c * stride] = 0.;
        for (unsigned int d = 0; d < dim; ++d)
          gradients[d][idx + c * stride] = 0.;
      }
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
    const unsigned int idx = compute_index<dim, n_q_points_1d>();
    const unsigned int offset =
      is_primitive ? tensor_dofs_per_cell : rt_tensor_dofs_per_component;
    const unsigned int stride = is_primitive ? offset : rt_tensor_dofs_per_cell;

    if (is_primitive)
      for (unsigned int c = 0; c < n_components_; ++c)
        {
          auto active_idx    = compute_active_index<dim, n_q_points_1d>();
          auto is_active_idx = is_active<dim, n_q_points_1d>();

          if (is_active_idx)
            {
              const dealii::types::global_dof_index src_idx =
                local_to_global[active_idx + c * offset];

              // Use the read-only data cache.
              values[idx + c * stride] = __ldg(&src[src_idx]);

              // dealii::CUDAWrappers::internal::
              // resolve_hanging_nodes<dim, fe_degree + 1, false>(
              //   constraint_mask[c], &values[c * stride]);
            }
        }
    else
      for (unsigned int c = 0; c < n_components_; ++c)
        {
          auto active_idx    = compute_active_index<dim, n_q_points_1d>(c);
          auto is_active_idx = is_active<dim, n_q_points_1d>(c);

          if (is_active_idx)
            {
              const dealii::types::global_dof_index src_idx =
                local_to_global[active_idx + c * offset];

              // Use the read-only data cache.
              values[idx + c * stride] = __ldg(&src[src_idx]);

              Number tmp = 0;
              for (auto it = constraint_range[0]; it < constraint_range[1];
                   ++it)
                if (src_idx == hanging_nodes_constraint[it])
                  {
                    tmp += src[hanging_nodes_constraint_indicator[it]] *
                           hanging_nodes_constraint_weights[it];
                    values[idx + c * stride] = tmp;
                  }
            }
        }

    // if (blockIdx.x == 0)
    //   printf("Read  %2d %d: %f, %f\n",
    //          idx,
    //          is_primitive,
    //          values[idx],
    //          values[idx + rt_tensor_dofs_per_cell]);


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
    const unsigned int idx = compute_index<dim, n_q_points_1d>();
    const unsigned int offset =
      is_primitive ? tensor_dofs_per_cell : rt_tensor_dofs_per_component;
    const unsigned int stride = is_primitive ? offset : rt_tensor_dofs_per_cell;

    if (is_primitive)
      for (unsigned int c = 0; c < n_components_; ++c)
        {
          auto active_idx    = compute_active_index<dim, n_q_points_1d>();
          auto is_active_idx = is_active<dim, n_q_points_1d>();

          if (is_active_idx)
            {
              const dealii::types::global_dof_index destination_idx =
                local_to_global[active_idx + c * offset];

              // dealii::CUDAWrappers::internal::
              //   resolve_hanging_nodes<dim, fe_degree + 1, true>(
              //     constraint_mask[c], &values[c * stride]);

              if (use_coloring)
                dst[destination_idx] += values[idx + c * stride];
              else
                atomicAdd(&dst[destination_idx], values[idx + c * stride]);
            }
        }
    else
      for (unsigned int c = 0; c < n_components_; ++c)
        {
          auto active_idx    = compute_active_index<dim, n_q_points_1d>(c);
          auto is_active_idx = is_active<dim, n_q_points_1d>(c);

          if (is_active_idx)
            {
              bool is_constrained = false;

              const dealii::types::global_dof_index destination_idx =
                local_to_global[active_idx + c * offset];

              for (auto it = constraint_range[0]; it < constraint_range[1];
                   ++it)
                if (destination_idx == hanging_nodes_constraint[it])
                  {
                    atomicAdd(&dst[hanging_nodes_constraint_indicator[it]],
                              values[idx + c * stride] *
                                hanging_nodes_constraint_weights[it]);

                    is_constrained = true;
                  }

              if (!is_constrained)
                {
                  if (use_coloring)
                    dst[destination_idx] += values[idx + c * stride];
                  else
                    atomicAdd(&dst[destination_idx], values[idx + c * stride]);
                }
            }
        }

    // if (blockIdx.x == 0)
    //   printf("Store %2d %d: %f, %f\n",
    //          idx,
    //          is_primitive,
    //          values[idx],
    //          values[idx + rt_tensor_dofs_per_cell]);

    // if (blockIdx.x == 0)
    //   printf("0-%d: %f, %f\n", idx, values[idx], values[idx + stride]);

    // if (blockIdx.x == 6)
    //   printf("2-%d: %f, %f\n", idx, values[idx], values[idx + stride]);
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
    if (is_primitive)
      {
        EvaluatorTensorProduct<EvaluatorVariant::evaluate_general,
                               dim,
                               fe_degree,
                               n_q_points_1d,
                               n_components_,
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
    else
      {
        EvaluatorTensorProduct<EvaluatorVariant::evaluate_raviart_thomas,
                               dim,
                               fe_degree,
                               n_q_points_1d,
                               n_components_,
                               Number>
          evaluator_tensor_product(mf_object_id);
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
    if (is_primitive)
      {
        EvaluatorTensorProduct<EvaluatorVariant::evaluate_general,
                               dim,
                               fe_degree,
                               n_q_points_1d,
                               n_components_,
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
            evaluator_tensor_product.integrate_gradient<false>(values,
                                                               gradients);
            __syncthreads();
          }

        // auto idx   = compute_index<dim, n_q_points_1d>();
        // auto shape = get_cell_shape_values<Number>(mf_object_id);

        // if (blockIdx.x == 0)
        //   printf("Int   %2d %d: %f, %f, %f\n",
        //          idx,
        //          is_primitive,
        //          values[idx],
        //          values[idx + rt_tensor_dofs_per_cell],
        //          shape[idx]);
      }
    else
      {
        EvaluatorTensorProduct<EvaluatorVariant::evaluate_raviart_thomas,
                               dim,
                               fe_degree,
                               n_q_points_1d,
                               n_components_,
                               Number>
          evaluator_tensor_product(mf_object_id);
        if (integrate_val == true && integrate_grad == true)
          {
            // todo
            // evaluator_tensor_product.integrate_value_and_gradient(values,
            //                                                       gradients);

            evaluator_tensor_product.integrate_value(values);
            __syncthreads();

            evaluator_tensor_product.integrate_gradient<true>(values,
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
            evaluator_tensor_product.integrate_gradient<false>(values,
                                                               gradients);
            __syncthreads();
          }
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

    value_type val = {};

    if (is_primitive)
      {
        for (unsigned int c = 0; c < n_components_; ++c)
          val[c] = values[q_point + c * n_q_points];
      }
    else
      {
        const Number *jacobian = &jac[q_point];

        for (unsigned int c = 0; c < n_components_; ++c)
          val[c] = values[q_point + c * n_q_points] *
                   jacobian[padding_length * n_cells * (dim * c + c)] *
                   inv_det[q_point];
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
    const unsigned int dof = compute_index<dim, n_q_points_1d>();

    value_type val = {};

    if (is_primitive)
      {
        for (unsigned int c = 0; c < n_components_; ++c)
          val[c] = values[dof + c * tensor_dofs_per_cell];
      }
    else // todo
      {
        for (unsigned int c = 0; c < n_components_; ++c)
          val[c] = values[dof + c * rt_tensor_dofs_per_cell];
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
    const unsigned int q_point = compute_index<dim, n_q_points_1d>();

    if (is_primitive)
      {
        for (unsigned int c = 0; c < n_components_; ++c)
          values[q_point + c * n_q_points] = val_in[c] * JxW[q_point];
      }
    else
      {
        const Number *jacobian = &jac[q_point];
        const Number  fac      = weights[q_point];

        for (unsigned int c = 0; c < n_components_; ++c)
          {
            values[q_point + c * n_q_points] =
              val_in[c] * jacobian[padding_length * n_cells * (dim * c + c)] *
              fac;
          }
      }
    __syncthreads();

    // if (blockIdx.x == 0)
    //   printf("Sub V %2d %d: %f, %f\n",
    //          q_point,
    //          is_primitive,
    //          values[q_point],
    //          values[q_point + rt_tensor_dofs_per_cell]);
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

    if (is_primitive)
      {
        for (unsigned int c = 0; c < n_components_; ++c)
          values[dof + c * tensor_dofs_per_cell] = val_in[c];
      }
    else // todo
      {
        for (unsigned int c = 0; c < n_components_; ++c)
          values[dof + c * rt_tensor_dofs_per_cell] = val_in[c];
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
    // TODO optimize if the mesh is uniform
    const unsigned int q_point      = compute_index<dim, n_q_points_1d>();
    const Number      *inv_jacobian = &inv_jac[q_point];
    gradient_type      grad         = {};

    if (is_primitive)
      {
        for (unsigned int c = 0; c < n_components_; ++c)
          for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
            {
              Number tmp = 0.;
              for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                tmp +=
                  inv_jacobian[padding_length * n_cells * (dim * d_2 + d_1)] *
                  gradients[d_2][q_point + c * n_q_points];
              grad[d_1][c] = tmp;
            }
      }
    else
      {
        // cartesian mesh only
        const Number *jacobian = &jac[q_point];

        // J * grad_quad * J^-1 * det(J^-1)
        for (unsigned int c = 0; c < n_components_; ++c)
          for (unsigned int d = 0; d < dim; ++d)
            {
              grad[d][c] =
                inv_jacobian[padding_length * n_cells * (dim * d + d)] *
                gradients[d][q_point + c * n_q_points] *
                jacobian[padding_length * n_cells * (dim * c + c)] *
                inv_det[q_point];
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
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    submit_gradient(const gradient_type &grad_in)
  {
    // TODO optimize if the mesh is uniform
    const unsigned int q_point      = compute_index<dim, n_q_points_1d>();
    const Number      *inv_jacobian = &inv_jac[q_point];

    if (is_primitive)
      {
        for (unsigned int c = 0; c < n_components_; ++c)
          for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
            {
              Number tmp = 0.;
              for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                tmp +=
                  inv_jacobian[n_cells * padding_length * (dim * d_1 + d_2)] *
                  grad_in[d_2][c];
              gradients[d_1][q_point + c * n_q_points] = tmp * JxW[q_point];
            }
      }
    else
      {
        // Cartesian cell
        const Number *jacobian = &jac[q_point];
        const Number  fac      = weights[q_point];

        for (unsigned int c = 0; c < n_components_; ++c)
          for (unsigned int d = 0; d < dim; ++d)
            {
              gradients[d][q_point + c * n_q_points] =
                inv_jacobian[n_cells * padding_length * (dim * d + d)] *
                jacobian[padding_length * n_cells * (dim * c + c)] * fac *
                grad_in[d][c];
            }
      }
    __syncthreads();
  }


  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ Number
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    get_divergence() const
  {
    const unsigned int q_point = compute_index<dim, n_q_points_1d>();

    Number div;

    div = gradients[0][q_point] * inv_det[q_point];

    for (unsigned int d = 1; d < dim; ++d)
      div += gradients[d][q_point + d * n_q_points] * inv_det[q_point];

    // if (blockIdx.x == 0)
    //   printf("Get D %2d %d: %f, %f, %f\n",
    //          q_point,
    //          is_primitive,
    //          gradients[0][q_point],
    //          gradients[1][q_point],
    //          div);

    return div;
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    submit_divergence(const Number &div_in)
  {
    const unsigned int q_point = compute_index<dim, n_q_points_1d>();

    const Number fac = weights[q_point] * div_in;

    for (unsigned int d = 0; d < dim; ++d)
      {
        gradients[d][q_point + d * n_q_points] = fac;
        for (unsigned int e = d + 1; e < dim; ++e)
          {
            gradients[d][q_point + e * n_q_points] = 0.;
            gradients[e][q_point + d * n_q_points] = 0.;
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
                     const data_type         &data,
                     SharedData<dim, Number> *shdata,
                     const bool               is_interior_face)
    : n_faces(data.n_faces)
    , n_cells(data.n_cells)
    , padding_length(data.padding_length)
    , face_padding_length(data.face_padding_length)
    , mf_object_id(data.id)
    , hanging_nodes_constraint(data.hanging_nodes_constraint)
    , hanging_nodes_constraint_indicator(
        data.hanging_nodes_constraint_indicator)
    , hanging_nodes_constraint_weights(data.hanging_nodes_constraint_weights)
    , hanging_nodes_constraint_coarse(data.hanging_nodes_constraint_coarse)
    , use_coloring(data.use_coloring)
    , is_interior_face(is_interior_face)
    , is_primitive(data.is_primitive)
    , matrix_type(data.matrix_type)
  {
    auto face_no = is_interior_face ? face_id : face_id + n_faces;

    cell_id = data.face2cell_id[face_no];

    constraint_range = data.constraint_range + cell_id * 2;
    constraint_coarse_range =
      data.constraint_coarse_range + data.face2cell_id[face_id] * 2;

    local_to_global =
      data.local_to_global + padding_length * n_components_ * cell_id;
    l_to_g_coarse =
      data.l_to_g_coarse + padding_length * n_components_ * cell_id;

    jac            = data.face_jacobian + face_padding_length * face_no;
    inv_jac        = data.face_inv_jacobian + face_padding_length * face_no;
    JxW            = data.face_JxW + face_padding_length * face_no;
    weights        = data.face_q_weights;
    inv_det        = data.face_inv_det + face_padding_length * face_no;
    normal_vec     = data.normal_vector + face_padding_length * face_no;
    face_number    = data.face_number[face_no];
    subface_number = data.subface_number[face_no];

    if (matrix_type == MatrixType::active_matrix)
      {
        ignore_read  = false;
        ignore_write = false;
      }
    else if (matrix_type == MatrixType::level_matrix)
      {
        ignore_read  = !is_interior_face && subface_number != -1;
        ignore_write = ignore_read;
      }
    else if (matrix_type == MatrixType::edge_down_matrix)
      {
        constraint_range =
          data.constraint_range + data.face2cell_id[face_id] * 2;

        ignore_read  = !is_interior_face || subface_number == -1;
        ignore_write = is_interior_face || subface_number == -1;
      }
    else if (matrix_type == MatrixType::edge_up_matrix)
      {
        constraint_range =
          data.constraint_range + data.face2cell_id[face_id] * 2;

        ignore_read  = is_interior_face || subface_number == -1;
        ignore_write = !is_interior_face || subface_number == -1;
      }
    else if (matrix_type == MatrixType::interface_matrix)
      {
        ignore_read  = !is_interior_face && subface_number != -1;
        ignore_write = !is_interior_face && subface_number != -1;
      }
    const unsigned int stride =
      is_primitive ? rt_tensor_dofs_per_cell : rt_tensor_dofs_per_cell;
    const unsigned int shift = is_interior_face ? 0 : n_components_ * stride;

    values = &shdata->values[shift];
    for (unsigned int i = 0; i < dim; ++i)
      gradients[i] = &shdata->gradients[i][shift];

    const unsigned int idx = compute_index<dim, n_q_points_1d>();
    for (unsigned int c = 0; c < n_components_; ++c)
      {
        values[idx + c * stride] = 0.;
        for (unsigned int d = 0; d < dim; ++d)
          gradients[d][idx + c * stride] = 0.;
      }
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
    const unsigned int idx = compute_index<dim, n_q_points_1d>();
    const unsigned int offset =
      is_primitive ? tensor_dofs_per_cell : rt_tensor_dofs_per_component;
    const unsigned int stride = is_primitive ? offset : rt_tensor_dofs_per_cell;

    if (is_primitive)
      for (unsigned int c = 0; c < n_components_; ++c)
        {
          const dealii::types::global_dof_index src_idx =
            local_to_global[idx + c * offset];

          if (ignore_read)
            values[idx + c * stride] = 0;
          else
            values[idx + c * stride] = __ldg(&src[src_idx]);
        }
    else
      for (unsigned int c = 0; c < n_components_; ++c)
        {
          if (c == face_number / 2)
            continue;

          auto active_idx    = compute_active_index<dim, n_q_points_1d>(c);
          auto is_active_idx = is_active<dim, n_q_points_1d>(c);

          if (is_active_idx)
            {
              const dealii::types::global_dof_index src_idx =
                local_to_global[active_idx + c * offset];

              if (ignore_read)
                values[idx + c * stride] = 0;
              else
                {
                  values[idx + c * stride] = __ldg(&src[src_idx]);

                  if (matrix_type == MatrixType::active_matrix ||
                      matrix_type == MatrixType::level_matrix)
                    {
                      Number tmp = 0;
                      for (auto it = constraint_range[0];
                           it < constraint_range[1];
                           ++it)
                        if (src_idx == hanging_nodes_constraint[it])
                          {
                            tmp += src[hanging_nodes_constraint_indicator[it]] *
                                   hanging_nodes_constraint_weights[it];
                            values[idx + c * stride] = tmp;
                          }
                    }
                }
            }
        }


    // if (blockIdx.x == 0)
    //   printf("0-%d: %f, %f\n", idx, values[idx], values[idx + stride]);

    // if (blockIdx.x == 3)
    //   printf("2-%d: %f, %f\n", idx, values[idx], values[idx + stride]);

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
    const unsigned int idx = compute_index<dim, n_q_points_1d>();
    const unsigned int offset =
      is_primitive ? tensor_dofs_per_cell : rt_tensor_dofs_per_component;
    const unsigned int stride = is_primitive ? offset : rt_tensor_dofs_per_cell;

    if (is_primitive)
      for (unsigned int c = 0; c < n_components_; ++c)
        {
          const dealii::types::global_dof_index destination_idx =
            l_to_g_coarse[idx + c * offset];

          if (use_coloring && !ignore_write)
            dst[destination_idx] += values[idx + c * stride];
          else if (!use_coloring && !ignore_write)
            atomicAdd(&dst[destination_idx], values[idx + c * stride]);
        }
    else
      for (unsigned int c = 0; c < n_components_; ++c)
        {
          if (c == face_number / 2)
            continue;

          auto active_idx    = compute_active_index<dim, n_q_points_1d>(c);
          auto is_active_idx = is_active<dim, n_q_points_1d>(c);

          if (is_active_idx)
            {
              const dealii::types::global_dof_index destination_idx =
                l_to_g_coarse[active_idx + c * offset];

              bool is_constrained = false;

              if (matrix_type == MatrixType::active_matrix ||
                  matrix_type == MatrixType::level_matrix)
                for (auto it = constraint_range[0]; it < constraint_range[1];
                     ++it)
                  if (!ignore_write &&
                      destination_idx == hanging_nodes_constraint[it])
                    {
                      atomicAdd(&dst[hanging_nodes_constraint_indicator[it]],
                                values[idx + c * stride] *
                                  hanging_nodes_constraint_weights[it]);

                      is_constrained = true;
                    }

              if (!is_constrained)
                {
                  if (use_coloring && !ignore_write)
                    dst[destination_idx] += values[idx + c * stride];
                  else if (!use_coloring && !ignore_write)
                    atomicAdd(&dst[destination_idx], values[idx + c * stride]);
                }
            }
        }

    // if (blockIdx.x == 0)
    //   printf("0-%d: %f, %f\n", idx, values[idx], values[idx + stride]);

    // if (blockIdx.x == 0)
    //   printf("0-%d: %f, %f\n", idx, values[idx], values[idx + stride]);
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
                           n_components_,
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
                           n_components_,
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
    const unsigned int q_point_face =
      compute_face_index<dim, n_q_points_1d>(face_number / 2);

    value_type val = {};

    if (is_primitive)
      {
        for (unsigned int c = 0; c < n_components_; ++c)
          val[c] = values[q_point + c * rt_tensor_dofs_per_cell];
      }
    else
      {
        const Number *jacobian = &jac[q_point_face];

        for (unsigned int c = 0; c < n_components_; ++c)
          val[c] = values[q_point + c * rt_tensor_dofs_per_cell] *
                   jacobian[face_padding_length * n_cells * (dim * c + c)] *
                   inv_det[q_point_face];
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
    const unsigned int dof = compute_index<dim, n_q_points_1d>();

    value_type val = {};

    if (is_primitive)
      {
        for (unsigned int c = 0; c < n_components_; ++c)
          val[c] = values[dof + c * tensor_dofs_per_cell];
      }
    else // todo
      {
        for (unsigned int c = 0; c < n_components_; ++c)
          val[c] = values[dof + c * rt_tensor_dofs_per_cell];
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
    const unsigned int q_point = compute_index<dim, n_q_points_1d>();
    const unsigned int q_point_face =
      compute_face_index<dim, n_q_points_1d>(face_number / 2);

    if (is_primitive)
      {
        for (unsigned int c = 0; c < n_components_; ++c)
          values[q_point + c * rt_tensor_dofs_per_cell] =
            val_in[c] * JxW[q_point_face];
      }
    else
      {
        const Number *jacobian = &jac[q_point_face];
        const Number  fac      = JxW[q_point_face] * inv_det[q_point_face];

        for (unsigned int c = 0; c < n_components_; ++c)
          {
            values[q_point + c * rt_tensor_dofs_per_cell] =
              val_in[c] *
              jacobian[face_padding_length * n_cells * (dim * c + c)] * fac;
          }
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
    const unsigned int dof = compute_index<dim, fe_degree + 1>();

    if (is_primitive)
      {
        for (unsigned int c = 0; c < n_components_; ++c)
          values[dof + c * tensor_dofs_per_cell] = val_in[c];
      }
    else // todo
      {
        for (unsigned int c = 0; c < n_components_; ++c)
          values[dof + c * rt_tensor_dofs_per_cell] = val_in[c];
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
    // TODO optimize if the mesh is uniform
    const unsigned int q_point = compute_index<dim, n_q_points_1d>();
    const unsigned int q_point_face =
      compute_face_index<dim, n_q_points_1d>(face_number / 2);
    const Number *inv_jacobian = &inv_jac[q_point_face];
    gradient_type grad         = {};

    if (is_primitive)
      {
        for (unsigned int c = 0; c < n_components_; ++c)
          for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
            {
              Number tmp = 0.;
              for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                tmp += inv_jacobian[n_cells * face_padding_length *
                                    (dim * d_2 + d_1)] *
                       gradients[d_2][q_point + c * rt_tensor_dofs_per_cell];
              grad[d_1][c] = tmp;
            }
      }
    else
      {
        // cartesian mesh only
        const Number *jacobian = &jac[q_point_face];

        // J * grad_quad * J^-1 * det(J^-1)
        for (unsigned int c = 0; c < n_components_; ++c)
          for (unsigned int d = 0; d < dim; ++d)
            {
              grad[d][c] =
                jacobian[face_padding_length * n_cells * (dim * c + c)] *
                inv_jacobian[face_padding_length * n_cells * (dim * d + d)] *
                gradients[d][q_point + c * rt_tensor_dofs_per_cell] *
                inv_det[q_point_face];
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
    // TODO optimize if the mesh is uniform
    const unsigned int q_point = compute_index<dim, n_q_points_1d>();
    const unsigned int q_point_face =
      compute_face_index<dim, n_q_points_1d>(face_number / 2);
    const Number *inv_jacobian = &inv_jac[q_point_face];

    if (is_primitive)
      {
        for (unsigned int c = 0; c < n_components_; ++c)
          for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
            {
              Number tmp = 0.;
              for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                tmp += inv_jacobian[n_cells * face_padding_length *
                                    (dim * d_1 + d_2)] *
                       grad_in[d_2][c];
              gradients[d_1][q_point + c * rt_tensor_dofs_per_cell] =
                tmp * JxW[q_point_face];
            }
      }
    else
      {
        // Cartesian cell
        const Number *jacobian = &jac[q_point_face];
        const Number  fac      = JxW[q_point_face] * inv_det[q_point_face];

        for (unsigned int c = 0; c < n_components_; ++c)
          for (unsigned int d = 0; d < dim; ++d)
            {
              gradients[d][q_point + c * rt_tensor_dofs_per_cell] =
                inv_jacobian[n_cells * face_padding_length * (dim * d + d)] *
                jacobian[face_padding_length * n_cells * (dim * c + c)] * fac *
                grad_in[d][c];
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
    // TODO optimize if the mesh is uniform
    const unsigned int q_point_face =
      compute_face_index<dim, n_q_points_1d>(face_number / 2);
    const Number *normal_vector = &normal_vec[q_point_face];

    gradient_type grad              = get_gradient();
    value_type    normal_derivative = {};

    const Number fac = is_interior_face ? 1. : -1.;

    for (unsigned int c = 0; c < n_components_; ++c)
      for (unsigned int d = 0; d < dim; ++d)
        normal_derivative[c] +=
          grad[d][c] * normal_vector[n_cells * face_padding_length * d] * fac;

    // for (unsigned int c = 0; c < n_components_; ++c)
    //   if (c == face_number / 2)

    // for (unsigned int d = 0; d < dim; ++d)
    //   if (d == face_number / 2)
    //     normal_derivative[d] = tmp[d];

    // if (blockIdx.x == 1)
    //   printf("grad: 0-%d: %f, %f, %f, %f\n", q_point_face,
    //     grad[0][0], grad[0][1], grad[1][0], grad[1][1]);

    // if (blockIdx.x == 0)
    //   printf("nd: 0-%d: %f, %f\n", q_point_face,
    //     tmp[0], tmp[1]);

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
    // TODO optimize if the mesh is uniform
    const unsigned int q_point = compute_index<dim, n_q_points_1d>();
    const unsigned int q_point_face =
      compute_face_index<dim, n_q_points_1d>(face_number / 2);
    const Number *normal_vector = &normal_vec[q_point_face];
    const Number *inv_jacobian  = &inv_jac[q_point_face];

    const Number coe =
      (dim == 2 ? 1 : inv_jacobian[0]) * (is_interior_face ? 1. : -1.);

    gradient_type normal_x_jacobian = {};
    for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
      {
        Number tmp = 0.;
        for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
          tmp +=
            inv_jacobian[n_cells * face_padding_length * (dim * d_1 + d_2)] *
            normal_vector[n_cells * face_padding_length * d_2];
        normal_x_jacobian[d_1][0] = coe * tmp;
      }

    const Number fac =
      JxW[q_point_face] * inv_det[q_point_face] * (is_interior_face ? 1. : -1.);

    for (unsigned int c = 0; c < n_components_; ++c)
      for (unsigned int d = 0; d < dim; ++d)
        gradients[d][q_point + c * rt_tensor_dofs_per_cell] =
          grad_in[c] * normal_vector[n_cells * face_padding_length * d] * fac;

    // if (blockIdx.x == 1)
    //   printf("grad: 0-%d: %f, %f, %f, %f\n", q_point_face,
    //     gradients[0][q_point], gradients[0][q_point + 1 *
    //     rt_tensor_dofs_per_cell], gradients[1][q_point], gradients[1][q_point
    //     + 1 * rt_tensor_dofs_per_cell]);

    // if (blockIdx.x == 0)
    //   printf("grad: 0-%d: %f, %f\n", q_point, grad_in[0], grad_in[1]);

    // if (blockIdx.x == 1)
    //   printf("grad: 1-%d: %f, %f\n", q_point, grad_in[0], grad_in[1]);

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