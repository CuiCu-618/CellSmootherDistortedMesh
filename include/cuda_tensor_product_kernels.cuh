/**
 * @file cuda_tensor_product_kernels.cuh
 * @brief Generic evaluator framework.
 *
 * Generic evaluator framework that valuates the given shape data in general
 * dimensions using the tensor product form. Depending on the particular layout
 * in the matrix entries, this corresponds to a usual matrix-matrix product or a
 * matrix-matrix product including some symmetries.
 *
 * @author Cu Cui
 * @date 2024-01-22
 * @version 0.1
 *
 * @remark
 * @note
 * @warning
 */


#ifndef CUDA_TENSOR_PRODUCT_KERNELS_CUH
#define CUDA_TENSOR_PRODUCT_KERNELS_CUH

#include <deal.II/base/config.h>

#include <deal.II/base/utilities.h>

#include "cuda_matrix_free.cuh"


#define CONFLICTFREE


namespace PSMF
{

#ifdef CONFLICTFREE
  // Function to calculate GCD of two numbers
  __device__ constexpr unsigned int
  gcd(unsigned int a, unsigned int b)
  {
    if (b == 0)
      return a;
    return gcd(b, a % b);
  }

  // Recursive template function to calculate LCM of two numbers
  template <int a, int b>
  struct LCM
  {
    static constexpr unsigned int value = (a * b) / gcd(a, b);
  };

  // Function to calculate the multiple of a number
  template <int n, int constant>
  __device__ constexpr unsigned int
  calculate_multiple()
  {
    // Calculate the multiple of n
    constexpr unsigned int multiple = LCM<n, constant>::value / n;

    return multiple;
  }
#endif

  /**
   * For face integral, compute the offset for a given subface.
   */
  template <int dim, int n_points_2d, int dir>
  __device__ inline unsigned int
  compute_subface_offset(unsigned int face_number, int subface_number)
  {
    if (subface_number == -1 || subface_number == -2)
      return 0;

    if (dim == 2)
      return n_points_2d * (subface_number + 1);

    if (dir == 0)
      return face_number / 2 == 1 ? n_points_2d * ((subface_number / 2) + 1) :
                                    n_points_2d * ((subface_number & 1) + 1);

    if (dir == 1)
      return face_number / 2 == 2 ? n_points_2d * ((subface_number / 2) + 1) :
                                    n_points_2d * ((subface_number & 1) + 1);

    if (dir == 2)
      return face_number / 2 == 1 ? n_points_2d * ((subface_number & 1) + 1) :
                                    n_points_2d * ((subface_number / 2) + 1);

    return 0;
  }


  /**
   * In this namespace, the evaluator routines that evaluate the tensor
   * products are implemented.
   *
   * @ingroup MatrixFree
   */
  // TODO: for now only the general variant and face are implemented
  enum EvaluatorVariant
  {
    /**
     * Do not use anything more than the tensor product structure of the finite
     * element.
     */
    evaluate_general,
    /**
     * Perform evaluation by exploiting symmetry in the finite element: i.e.,
     * skip some computations by utilizing the symmetry in the shape functions
     * and quadrature points.
     */
    evaluate_symmetric,
    /**
     * Raviart-Thomas elements with anisotropic polynomials.
     */
    evaluate_raviart_thomas,
    /**
     * Tensor product structure on faces.
     */
    evaluate_face
  };

  /**
   * Generic evaluator framework.
   *
   * @ingroup MatrixFree
   */
  template <EvaluatorVariant variant,
            int              dim,
            int              fe_degree,
            int              n_q_points_1d,
            typename Number>
  struct EvaluatorTensorProduct
  {
    const int mf_object_id;
  };

  /**
   * Internal evaluator for 1d-3d shape function using the tensor product form
   * of the basis functions.
   *
   * @ingroup MatrixFree
   */
  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  struct EvaluatorTensorProduct<evaluate_general,
                                dim,
                                fe_degree,
                                n_q_points_1d,
                                Number>
  {
    static constexpr unsigned int dofs_per_cell =
      dealii::Utilities::pow(fe_degree + 1, dim);
    static constexpr unsigned int n_q_points =
      dealii::Utilities::pow(n_q_points_1d, dim);

    __device__
    EvaluatorTensorProduct(int mf_object_id);

    __device__
    EvaluatorTensorProduct(int mf_object_id, Number *shv, Number *shg);

    /**
     * Evaluate the values of a finite element function at the quadrature
     * points.
     */
    template <int direction, bool dof_to_quad, bool add, bool in_place>
    __device__ void
    values(Number shape_values[], const Number *in, Number *out) const;

    /**
     * Evaluate the gradient of a finite element function at the quadrature
     * points for a given @p direction.
     */
    template <int direction, bool dof_to_quad, bool add, bool in_place>
    __device__ void
    gradients(Number shape_gradients[], const Number *in, Number *out) const;

    /**
     * Helper function for values() and gradients().
     */
    template <int direction, bool dof_to_quad, bool add, bool in_place>
    __device__ void
    apply(Number shape_data[], const Number *in, Number *out) const;

    /**
     * Evaluate the finite element function at the quadrature points.
     */
    __device__ void
    value_at_quad_pts(Number *u);

    /**
     * Helper function for integrate(). Integrate the finite element function.
     */
    __device__ void
    integrate_value(Number *u);

    /**
     * Evaluate the gradients of the finite element function at the quadrature
     * points.
     */
    __device__ void
    gradient_at_quad_pts(const Number *const u, Number *grad_u[dim]);

    /**
     * Evaluate the diagnoal of hessian of the finite element function at the
     * quadrature points.
     */
    __device__ void
    hessian_at_quad_pts(const Number *const u, Number *grad_u[dim]);

    /**
     * Evaluate the values and the gradients of the finite element function at
     * the quadrature points.
     */
    __device__ void
    value_and_gradient_at_quad_pts(Number *const u, Number *grad_u[dim]);

    /**
     * Helper function for integrate(). Integrate the gradients of the finite
     * element function.
     */
    template <bool add>
    __device__ void
    integrate_gradient(Number *u, Number *grad_u[dim]);

    /**
     * Helper function for integrate(). Integrate the values and the gradients
     * of the finite element function.
     */
    __device__ void
    integrate_value_and_gradient(Number *u, Number *grad_u[dim]);

    const int mf_object_id;

    Number *shape_values;
    Number *shape_gradients;
  };


  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  __device__
  EvaluatorTensorProduct<evaluate_general,
                         dim,
                         fe_degree,
                         n_q_points_1d,
                         Number>::EvaluatorTensorProduct(int object_id)
    : mf_object_id(object_id)
  {}

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  __device__
  EvaluatorTensorProduct<evaluate_general,
                         dim,
                         fe_degree,
                         n_q_points_1d,
                         Number>::EvaluatorTensorProduct(int     object_id,
                                                         Number *shv,
                                                         Number *shg)
    : mf_object_id(object_id)
    , shape_values(shv)
    , shape_gradients(shg)
  {}

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  template <int direction, bool dof_to_quad, bool add, bool in_place>
  __device__ void
  EvaluatorTensorProduct<evaluate_general,
                         dim,
                         fe_degree,
                         n_q_points_1d,
                         Number>::values(Number        shape_values[],
                                         const Number *in,
                                         Number       *out) const
  {
    apply<direction, dof_to_quad, add, in_place>(shape_values, in, out);
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  template <int direction, bool dof_to_quad, bool add, bool in_place>
  __device__ void
  EvaluatorTensorProduct<evaluate_general,
                         dim,
                         fe_degree,
                         n_q_points_1d,
                         Number>::gradients(Number        shape_gradients[],
                                            const Number *in,
                                            Number       *out) const
  {
    apply<direction, dof_to_quad, add, in_place>(shape_gradients, in, out);
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  template <int direction, bool dof_to_quad, bool add, bool in_place>
  __device__ void
  EvaluatorTensorProduct<evaluate_general,
                         dim,
                         fe_degree,
                         n_q_points_1d,
                         Number>::apply(Number        shape_data[],
                                        const Number *in,
                                        Number       *out) const
  {
#ifdef CONFLICTFREE
    constexpr unsigned int multiple = calculate_multiple<n_q_points_1d, 16>();

    const unsigned int z = dim == 3 ? threadIdx.z % n_q_points_1d : threadIdx.z;
    const unsigned int row =
      dim == 2 ? threadIdx.y % n_q_points_1d : threadIdx.y;
    const unsigned int col = threadIdx.x;

    Number t = 0;
    for (unsigned int k = 0; k < n_q_points_1d; ++k)
      {
        const unsigned int shape_idx =
          dof_to_quad ?
            ((direction == 0) ?
               (col + ((k + col / multiple) % n_q_points_1d) * n_q_points_1d) :
             (direction == 1) ? (row + k * n_q_points_1d) :
                                (z + k * n_q_points_1d)) :
            ((direction == 0) ?
               ((k + col / multiple) % n_q_points_1d + col * n_q_points_1d) :
             (direction == 1) ? (k + row * n_q_points_1d) :
                                (k + z * n_q_points_1d));
        const unsigned int source_idx =
          (direction == 0) ? ((k + col / multiple) % n_q_points_1d +
                              n_q_points_1d * (row + n_q_points_1d * z)) :
          (direction == 1) ? (col + n_q_points_1d * (k + n_q_points_1d * z)) :
                             (col + n_q_points_1d * (row + n_q_points_1d * k));
        t +=
          shape_data[shape_idx] * (in_place ? out[source_idx] : in[source_idx]);
      }

    if (in_place)
      __syncthreads();

    const unsigned int destination_idx =
      col + n_q_points_1d * (row + n_q_points_1d * z);

    if (add)
      out[destination_idx] += t;
    else
      out[destination_idx] = t;

#else
    const unsigned int z = dim == 3 ? threadIdx.z % n_q_points_1d : threadIdx.z;
    const unsigned int row =
      dim == 2 ? threadIdx.y % n_q_points_1d : threadIdx.y;
    const unsigned int col = threadIdx.x;

    Number t = 0;
    for (unsigned int k = 0; k < n_q_points_1d; ++k)
      {
        const unsigned int shape_idx =
          dof_to_quad ? (row + k * n_q_points_1d) : (k + row * n_q_points_1d);
        const unsigned int source_idx =
          (direction == 0) ? (k + n_q_points_1d * (col + n_q_points_1d * z)) :
          (direction == 1) ? (col + n_q_points_1d * (k + n_q_points_1d * z)) :
                             (col + n_q_points_1d * (z + n_q_points_1d * k));
        t +=
          shape_data[shape_idx] * (in_place ? out[source_idx] : in[source_idx]);
      }

    if (in_place)
      __syncthreads();

    const unsigned int destination_idx =
      (direction == 0) ? (row + n_q_points_1d * (col + n_q_points_1d * z)) :
      (direction == 1) ? (col + n_q_points_1d * (row + n_q_points_1d * z)) :
                         (col + n_q_points_1d * (z + n_q_points_1d * row));

    if (add)
      out[destination_idx] += t;
    else
      out[destination_idx] = t;
#endif
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  inline __device__ void
  EvaluatorTensorProduct<evaluate_general,
                         dim,
                         fe_degree,
                         n_q_points_1d,
                         Number>::value_at_quad_pts(Number *u)
  {
    switch (dim)
      {
        case 1:
          {
            values<0, true, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);

            break;
          }
        case 2:
          {
            values<0, true, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);
            __syncthreads();
            values<1, true, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);

            break;
          }
        case 3:
          {
            values<0, true, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);
            __syncthreads();
            values<1, true, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);
            __syncthreads();
            values<2, true, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);

            break;
          }
        default:
          {
            // Do nothing. We should throw but we can't from a __device__
            // function.
            printf(
              "Error: Invalid dimension. In file cuda_tensor_product_kernels.cuh:301\n");
          }
      }
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  inline __device__ void
  EvaluatorTensorProduct<evaluate_general,
                         dim,
                         fe_degree,
                         n_q_points_1d,
                         Number>::integrate_value(Number *u)
  {
    switch (dim)
      {
        case 1:
          {
            values<0, false, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);

            break;
          }
        case 2:
          {
            values<0, false, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);
            __syncthreads();
            values<1, false, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);

            break;
          }
        case 3:
          {
            values<0, false, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);
            __syncthreads();
            values<1, false, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);
            __syncthreads();
            values<2, false, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);

            break;
          }
        default:
          {
            // Do nothing. We should throw but we can't from a __device__
            // function.
            printf(
              "Error: Invalid dimension. In file cuda_tensor_product_kernels.cuh:353\n");
          }
      }
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  inline __device__ void
  EvaluatorTensorProduct<evaluate_general,
                         dim,
                         fe_degree,
                         n_q_points_1d,
                         Number>::gradient_at_quad_pts(const Number *const u,
                                                       Number *grad_u[dim])
  {
    switch (dim)
      {
        case 1:
          {
            gradients<0, true, false, false>(shape_values, u, grad_u[0]);

            break;
          }
        case 2:
          {
            gradients<0, true, false, false>(shape_gradients, u, grad_u[0]);
            values<0, true, false, false>(shape_values, u, grad_u[1]);

            __syncthreads();

            values<1, true, false, true>(shape_values, grad_u[0], grad_u[0]);
            gradients<1, true, false, true>(shape_gradients,
                                            grad_u[1],
                                            grad_u[1]);

            break;
          }
        case 3:
          {
            gradients<0, true, false, false>(shape_gradients, u, grad_u[0]);
            values<0, true, false, false>(shape_values, u, grad_u[1]);
            values<0, true, false, false>(shape_values, u, grad_u[2]);

            __syncthreads();

            values<1, true, false, true>(shape_values, grad_u[0], grad_u[0]);
            gradients<1, true, false, true>(shape_gradients,
                                            grad_u[1],
                                            grad_u[1]);
            values<1, true, false, true>(shape_values, grad_u[2], grad_u[2]);

            __syncthreads();

            values<2, true, false, true>(shape_values, grad_u[0], grad_u[0]);
            values<2, true, false, true>(shape_values, grad_u[1], grad_u[1]);
            gradients<2, true, false, true>(shape_gradients,
                                            grad_u[2],
                                            grad_u[2]);

            break;
          }
        default:
          {
            // Do nothing. We should throw but we can't from a __device__
            // function.
            printf(
              "Error: Invalid dimension. In file cuda_tensor_product_kernels.cuh:444\n");
          }
      }
  }


  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  inline __device__ void
  EvaluatorTensorProduct<evaluate_general,
                         dim,
                         fe_degree,
                         n_q_points_1d,
                         Number>::hessian_at_quad_pts(const Number *const u,
                                                      Number *grad_u[dim])
  {
    switch (dim)
      {
        case 1:
          {
            gradients<0, true, false, false>(
              get_cell_shape_hessians<Number>(mf_object_id), u, grad_u[0]);

            break;
          }
        case 2:
          {
            gradients<0, true, false, false>(
              get_cell_shape_hessians<Number>(mf_object_id), u, grad_u[0]);
            values<0, true, false, false>(
              get_cell_shape_values<Number>(mf_object_id), u, grad_u[1]);

            __syncthreads();

            values<1, true, false, true>(get_cell_shape_values<Number>(
                                           mf_object_id),
                                         grad_u[0],
                                         grad_u[0]);
            gradients<1, true, false, true>(get_cell_shape_hessians<Number>(
                                              mf_object_id),
                                            grad_u[1],
                                            grad_u[1]);

            break;
          }
        case 3:
          {
            gradients<0, true, false, false>(
              get_cell_shape_hessians<Number>(mf_object_id), u, grad_u[0]);
            values<0, true, false, false>(
              get_cell_shape_values<Number>(mf_object_id), u, grad_u[1]);
            values<0, true, false, false>(
              get_cell_shape_values<Number>(mf_object_id), u, grad_u[2]);

            __syncthreads();

            values<1, true, false, true>(get_cell_shape_values<Number>(
                                           mf_object_id),
                                         grad_u[0],
                                         grad_u[0]);
            gradients<1, true, false, true>(get_cell_shape_hessians<Number>(
                                              mf_object_id),
                                            grad_u[1],
                                            grad_u[1]);
            values<1, true, false, true>(get_cell_shape_values<Number>(
                                           mf_object_id),
                                         grad_u[2],
                                         grad_u[2]);

            __syncthreads();

            values<2, true, false, true>(get_cell_shape_values<Number>(
                                           mf_object_id),
                                         grad_u[0],
                                         grad_u[0]);
            values<2, true, false, true>(get_cell_shape_values<Number>(
                                           mf_object_id),
                                         grad_u[1],
                                         grad_u[1]);
            gradients<2, true, false, true>(get_cell_shape_hessians<Number>(
                                              mf_object_id),
                                            grad_u[2],
                                            grad_u[2]);

            break;
          }
        default:
          {
            // Do nothing. We should throw but we can't from a __device__
            // function.
            printf(
              "Error: Invalid dimension. In file cuda_tensor_product_kernels.cuh:444\n");
          }
      }
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  inline __device__ void
  EvaluatorTensorProduct<
    evaluate_general,
    dim,
    fe_degree,
    n_q_points_1d,
    Number>::value_and_gradient_at_quad_pts(Number *const u,
                                            Number       *grad_u[dim])
  {
    switch (dim)
      {
        case 1:
          {
            values<0, true, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);
            __syncthreads();

            gradients<0, true, false, false>(
              get_cell_co_shape_gradients<Number>(mf_object_id), u, grad_u[0]);

            break;
          }
        case 2:
          {
            values<0, true, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);
            __syncthreads();
            values<1, true, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);
            __syncthreads();

            gradients<0, true, false, false>(
              get_cell_co_shape_gradients<Number>(mf_object_id), u, grad_u[0]);
            gradients<1, true, false, false>(
              get_cell_co_shape_gradients<Number>(mf_object_id), u, grad_u[1]);

            break;
          }
        case 3:
          {
            values<0, true, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);
            __syncthreads();
            values<1, true, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);
            __syncthreads();
            values<2, true, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);
            __syncthreads();

            gradients<0, true, false, false>(
              get_cell_co_shape_gradients<Number>(mf_object_id), u, grad_u[0]);
            gradients<1, true, false, false>(
              get_cell_co_shape_gradients<Number>(mf_object_id), u, grad_u[1]);
            gradients<2, true, false, false>(
              get_cell_co_shape_gradients<Number>(mf_object_id), u, grad_u[2]);

            break;
          }
        default:
          {
            // Do nothing. We should throw but we can't from a __device__
            // function.
            printf(
              "Error: Invalid dimension. In file cuda_tensor_product_kernels.cuh:516\n");
          }
      }
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  template <bool add>
  inline __device__ void
  EvaluatorTensorProduct<evaluate_general,
                         dim,
                         fe_degree,
                         n_q_points_1d,
                         Number>::integrate_gradient(Number *u,
                                                     Number *grad_u[dim])
  {
    switch (dim)
      {
        case 1:
          {
            gradients<0, false, add, false>(shape_gradients, grad_u[dim], u);

            break;
          }
        case 2:
          {
            gradients<0, false, false, true>(shape_gradients,
                                             grad_u[0],
                                             grad_u[0]);
            values<0, false, false, true>(get_cell_shape_values<Number>(
                                            mf_object_id),
                                          grad_u[1],
                                          grad_u[1]);

            __syncthreads();

            values<1, false, add, false>(shape_values, grad_u[0], u);
            __syncthreads();
            gradients<1, false, true, false>(shape_gradients, grad_u[1], u);

            break;
          }
        case 3:
          {
            gradients<0, false, false, true>(shape_gradients,
                                             grad_u[0],
                                             grad_u[0]);
            values<0, false, false, true>(shape_values, grad_u[1], grad_u[1]);
            values<0, false, false, true>(shape_values, grad_u[2], grad_u[2]);

            __syncthreads();

            values<1, false, false, true>(shape_values, grad_u[0], grad_u[0]);
            gradients<1, false, false, true>(shape_gradients,
                                             grad_u[1],
                                             grad_u[1]);
            values<1, false, false, true>(shape_values, grad_u[2], grad_u[2]);

            __syncthreads();

            values<2, false, add, false>(shape_values, grad_u[0], u);
            __syncthreads();
            values<2, false, true, false>(shape_values, grad_u[1], u);
            __syncthreads();
            gradients<2, false, true, false>(shape_gradients, grad_u[2], u);

            break;
          }
        default:
          {
            // Do nothing. We should throw but we can't from a __device__
            // function.
            printf(
              "Error: Invalid dimension. In file cuda_tensor_product_kernels.cuh:611\n");
          }
      }
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  inline __device__ void
  EvaluatorTensorProduct<evaluate_general,
                         dim,
                         fe_degree,
                         n_q_points_1d,
                         Number>::integrate_value_and_gradient(Number *u,
                                                               Number
                                                                 *grad_u[dim])
  {
    switch (dim)
      {
        case 1:
          {
            gradients<0, false, true, false>(
              get_cell_co_shape_gradients<Number>(mf_object_id), grad_u[0], u);
            __syncthreads();

            values<0, false, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);

            break;
          }
        case 2:
          {
            gradients<1, false, true, false>(
              get_cell_co_shape_gradients<Number>(mf_object_id), grad_u[1], u);
            __syncthreads();
            gradients<0, false, true, false>(
              get_cell_co_shape_gradients<Number>(mf_object_id), grad_u[0], u);
            __syncthreads();

            values<1, false, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);
            __syncthreads();
            values<0, false, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);
            __syncthreads();

            break;
          }
        case 3:
          {
            gradients<2, false, true, false>(
              get_cell_co_shape_gradients<Number>(mf_object_id), grad_u[2], u);
            __syncthreads();
            gradients<1, false, true, false>(
              get_cell_co_shape_gradients<Number>(mf_object_id), grad_u[1], u);
            __syncthreads();
            gradients<0, false, true, false>(
              get_cell_co_shape_gradients<Number>(mf_object_id), grad_u[0], u);
            __syncthreads();

            values<2, false, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);
            __syncthreads();
            values<1, false, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);
            __syncthreads();
            values<0, false, false, true>(
              get_cell_shape_values<Number>(mf_object_id), u, u);
            __syncthreads();

            break;
          }
        default:
          {
            // Do nothing. We should throw but we can't from a __device__
            // function.
            printf(
              "Error: Invalid dimension. In file cuda_tensor_product_kernels.cuh:688\n");
          }
      }
  }


  /**
   * Internal evaluator for 1d-3d shape function using the tensor product form
   * of the basis functions, including face integral.
   *
   * @ingroup MatrixFree
   */
  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  struct EvaluatorTensorProduct<evaluate_face,
                                dim,
                                fe_degree,
                                n_q_points_1d,
                                Number>
  {
    static constexpr unsigned int dofs_per_cell =
      dealii::Utilities::pow(fe_degree + 1, dim);

    __device__
    EvaluatorTensorProduct(int mf_object_id,
                           int face_number,
                           int subface_number);

    __device__
    EvaluatorTensorProduct(int     mf_object_id,
                           int     face_number,
                           int     subface_number,
                           Number *shv,
                           Number *shg);

    /**
     * Evaluate the values of a finite element function at the quadrature
     * points.
     */
    template <int direction, bool dof_to_quad, bool add, bool in_place>
    __device__ void
    values(Number shape_values[], const Number *in, Number *out) const;

    /**
     * Evaluate the gradient of a finite element function at the quadrature
     * points for a given @p direction.
     */
    template <int direction, bool dof_to_quad, bool add, bool in_place>
    __device__ void
    gradients(Number shape_gradients[], const Number *in, Number *out) const;

    /**
     * Helper function for values() and gradients().
     */
    template <int direction, bool dof_to_quad, bool add, bool in_place>
    __device__ void
    apply(Number shape_data[], const Number *in, Number *out) const;

    /**
     * Evaluate the finite element function at the quadrature points.
     */
    __device__ void
    value_at_quad_pts(Number *u);

    /**
     * Helper function for integrate(). Integrate the finite element function.
     */
    __device__ void
    integrate_value(Number *u);

    /**
     * Evaluate the gradients of the finite element function at the quadrature
     * points.
     */
    __device__ void
    gradient_at_quad_pts(const Number *const u, Number *grad_u[dim]);

    /**
     * Evaluate the values and the gradients of the finite element function at
     * the quadrature points.
     */
    __device__ void
    value_and_gradient_at_quad_pts(Number *const u, Number *grad_u[dim]);

    /**
     * Helper function for integrate(). Integrate the gradients of the finite
     * element function.
     */
    template <bool add>
    __device__ void
    integrate_gradient(Number *u, Number *grad_u[dim]);

    /**
     * Helper function for integrate(). Integrate the values and the gradients
     * of the finite element function.
     */
    __device__ void
    integrate_value_and_gradient(Number *u, Number *grad_u[dim]);

    const int mf_object_id;
    const int face_number;
    const int subface_number;

    Number *shape_values;
    Number *shape_gradients;
  };


  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  __device__
  EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::
    EvaluatorTensorProduct(int object_id, int face_number, int subface_number)
    : mf_object_id(object_id)
    , face_number(face_number)
    , subface_number(subface_number)
  {}

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  __device__
  EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::
    EvaluatorTensorProduct(int     object_id,
                           int     face_number,
                           int     subface_number,
                           Number *shv,
                           Number *shg)
    : mf_object_id(object_id)
    , face_number(face_number)
    , subface_number(subface_number)
    , shape_values(shv)
    , shape_gradients(shg)
  {}

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  template <int direction, bool dof_to_quad, bool add, bool in_place>
  __device__ void
  EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::
    values(Number shape_values[], const Number *in, Number *out) const
  {
    apply<direction, dof_to_quad, add, in_place>(shape_values, in, out);
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  template <int direction, bool dof_to_quad, bool add, bool in_place>
  __device__ void
  EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::
    gradients(Number shape_gradients[], const Number *in, Number *out) const
  {
    apply<direction, dof_to_quad, add, in_place>(shape_gradients, in, out);
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  template <int direction, bool dof_to_quad, bool add, bool in_place>
  __device__ void
  EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::
    apply(Number shape_data[], const Number *in, Number *out) const
  {
#ifdef CONFLICTFREE
    constexpr unsigned int multiple = calculate_multiple<n_q_points_1d, 16>();

    const unsigned int z = dim == 3 ? threadIdx.z % n_q_points_1d : threadIdx.z;
    const unsigned int row =
      dim == 2 ? threadIdx.y % n_q_points_1d : threadIdx.y;
    const unsigned int col = threadIdx.x;

    Number t = 0;
    for (unsigned int k = 0; k < n_q_points_1d; ++k)
      {
        const unsigned int shape_idx =
          dof_to_quad ?
            ((direction == 0) ?
               (col + ((k + col / multiple) % n_q_points_1d) * n_q_points_1d) :
             (direction == 1) ? (row + k * n_q_points_1d) :
                                (z + k * n_q_points_1d)) :
            ((direction == 0) ?
               ((k + col / multiple) % n_q_points_1d + col * n_q_points_1d) :
             (direction == 1) ? (k + row * n_q_points_1d) :
                                (k + z * n_q_points_1d));
        const unsigned int source_idx =
          (direction == 0) ? ((k + col / multiple) % n_q_points_1d +
                              n_q_points_1d * (row + n_q_points_1d * z)) :
          (direction == 1) ? (col + n_q_points_1d * (k + n_q_points_1d * z)) :
                             (col + n_q_points_1d * (row + n_q_points_1d * k));
        t +=
          shape_data[shape_idx] * (in_place ? out[source_idx] : in[source_idx]);
      }

    if (in_place)
      __syncthreads();

    const unsigned int destination_idx =
      col + n_q_points_1d * (row + n_q_points_1d * z);

    if (add)
      out[destination_idx] += t;
    else
      out[destination_idx] = t;

#else
    const unsigned int z = dim == 3 ? threadIdx.z % n_q_points_1d : threadIdx.z;
    const unsigned int row =
      dim == 2 ? threadIdx.y % n_q_points_1d : threadIdx.y;
    const unsigned int col = threadIdx.x;

    Number t = 0;
    for (unsigned int k = 0; k < n_q_points_1d; ++k)
      {
        const unsigned int shape_idx =
          dof_to_quad ? (row + k * n_q_points_1d) : (k + row * n_q_points_1d);
        const unsigned int source_idx =
          (direction == 0) ? (k + n_q_points_1d * (col + n_q_points_1d * z)) :
          (direction == 1) ? (col + n_q_points_1d * (k + n_q_points_1d * z)) :
                             (col + n_q_points_1d * (z + n_q_points_1d * k));
        t +=
          shape_data[shape_idx] * (in_place ? out[source_idx] : in[source_idx]);
      }

    if (in_place)
      __syncthreads();

    const unsigned int destination_idx =
      (direction == 0) ? (row + n_q_points_1d * (col + n_q_points_1d * z)) :
      (direction == 1) ? (col + n_q_points_1d * (row + n_q_points_1d * z)) :
                         (col + n_q_points_1d * (z + n_q_points_1d * row));

    if (add)
      out[destination_idx] += t;
    else
      out[destination_idx] = t;
#endif
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  inline __device__ void
  EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::
    value_at_quad_pts(Number *u)
  {
    constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;

    const unsigned int shift = (face_number & 1) * n_q_points_2d;

    // // local refinement
    // const unsigned int offset0 =
    //   compute_subface_offset<dim, n_q_points_2d, 0>(face_number,
    //                                                 subface_number);
    // const unsigned int offset1 =
    //   compute_subface_offset<dim, n_q_points_2d, 1>(face_number,
    //                                                 subface_number);
    // const unsigned int offset2 =
    //   compute_subface_offset<dim, n_q_points_2d, 2>(face_number,
    //                                                 subface_number);


    Number *shape_value_dir0 =
      face_number / 2 == 0 ?
#if MEMORY_TYPE == 1
        get_face_shape_values<Number>(mf_object_id) + shift :
#else
        shape_values + n_q_points_2d + shift :
#endif
        shape_values; // + offset0;

    Number *shape_value_dir1 =
      face_number / 2 == 1 ?
#if MEMORY_TYPE == 1
        get_face_shape_values<Number>(mf_object_id) + shift :
#else
        shape_values + n_q_points_2d + shift :
#endif
        shape_values; // + offset1;

    Number *shape_value_dir2 =
      face_number / 2 == 2 ?
#if MEMORY_TYPE == 1
        get_face_shape_values<Number>(mf_object_id) + shift :
#else
        shape_values + n_q_points_2d + shift :
#endif
        shape_values; // + offset2;

    switch (dim)
      {
        case 1:
          {
            values<0, true, false, true>(shape_value_dir0, u, u);

            break;
          }
        case 2:
          {
            values<0, true, false, true>(shape_value_dir0, u, u);
            __syncthreads();
            values<1, true, false, true>(shape_value_dir1, u, u);

            break;
          }
        case 3:
          {
            values<0, true, false, true>(shape_value_dir0, u, u);
            __syncthreads();
            values<1, true, false, true>(shape_value_dir1, u, u);
            __syncthreads();
            values<2, true, false, true>(shape_value_dir2, u, u);

            break;
          }
        default:
          {
            // Do nothing. We should throw but we can't from a __device__
            // function.
            printf(
              "Error: Invalid dimension. In file cuda_tensor_product_kernels.cuh: \n"
              "EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::"
              "value_at_quad_pts().\n");
          }
      }
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  inline __device__ void
  EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::
    integrate_value(Number *u)
  {
    constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;

    const unsigned int shift = (face_number & 1) * n_q_points_2d;

    // // local refinement
    // const unsigned int offset0 =
    //   compute_subface_offset<dim, n_q_points_2d, 0>(face_number,
    //                                                 subface_number);
    // const unsigned int offset1 =
    //   compute_subface_offset<dim, n_q_points_2d, 1>(face_number,
    //                                                 subface_number);
    // const unsigned int offset2 =
    //   compute_subface_offset<dim, n_q_points_2d, 2>(face_number,
    //                                                 subface_number);

    Number *shape_value_dir0 =
      face_number / 2 == 0 ?
#if MEMORY_TYPE == 1
        get_face_shape_values<Number>(mf_object_id) + shift :
#else
        shape_values + n_q_points_2d + shift :
#endif
        shape_values; // + offset0;

    Number *shape_value_dir1 =
      face_number / 2 == 1 ?
#if MEMORY_TYPE == 1
        get_face_shape_values<Number>(mf_object_id) + shift :
#else
        shape_values + n_q_points_2d + shift :
#endif
        shape_values; // + offset1;

    Number *shape_value_dir2 =
      face_number / 2 == 2 ?
#if MEMORY_TYPE == 1
        get_face_shape_values<Number>(mf_object_id) + shift :
#else
        shape_values + n_q_points_2d + shift :
#endif
        shape_values; // + offset2;

    switch (dim)
      {
        case 1:
          {
            values<0, false, false, true>(shape_value_dir0, u, u);

            break;
          }
        case 2:
          {
            values<0, false, false, true>(shape_value_dir0, u, u);
            __syncthreads();
            values<1, false, false, true>(shape_value_dir1, u, u);

            break;
          }
        case 3:
          {
            values<0, false, false, true>(shape_value_dir0, u, u);
            __syncthreads();
            values<1, false, false, true>(shape_value_dir1, u, u);
            __syncthreads();
            values<2, false, false, true>(shape_value_dir2, u, u);

            break;
          }
        default:
          {
            // Do nothing. We should throw but we can't from a __device__
            // function.
            printf(
              "Error: Invalid dimension.\n In file cuda_tensor_product_kernels.cuh: "
              "EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::"
              "integrate_value().\n");
          }
      }
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  inline __device__ void
  EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::
    gradient_at_quad_pts(const Number *const u, Number *grad_u[dim])
  {
    constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;

    const unsigned int shift = (face_number & 1) * n_q_points_2d;

    // // local refinement
    // const unsigned int offset0 =
    //   compute_subface_offset<dim, n_q_points_2d, 0>(face_number,
    //                                                 subface_number);
    // const unsigned int offset1 =
    //   compute_subface_offset<dim, n_q_points_2d, 1>(face_number,
    //                                                 subface_number);
    // const unsigned int offset2 =
    //   compute_subface_offset<dim, n_q_points_2d, 2>(face_number,
    //                                                 subface_number);


    Number *shape_value_dir0 =
      face_number / 2 == 0 ?
#if MEMORY_TYPE == 1
        get_face_shape_values<Number>(mf_object_id) + shift :
#else
        shape_values + n_q_points_2d + shift :
#endif
        shape_values; // + offset0;

    Number *shape_value_dir1 =
      face_number / 2 == 1 ?
#if MEMORY_TYPE == 1
        get_face_shape_values<Number>(mf_object_id) + shift :
#else
        shape_values + n_q_points_2d + shift :
#endif
        shape_values; // + offset1;

    Number *shape_value_dir2 =
      face_number / 2 == 2 ?
#if MEMORY_TYPE == 1
        get_face_shape_values<Number>(mf_object_id) + shift :
#else
        shape_values + n_q_points_2d + shift :
#endif
        shape_values; // + offset2;

    Number *shape_gradient_dir0 =
      face_number / 2 == 0 ?
#if MEMORY_TYPE == 1
        get_face_shape_gradients<Number>(mf_object_id) + shift :
#else
        shape_gradients + n_q_points_2d + shift :
#endif
        shape_gradients; // + offset0;

    Number *shape_gradient_dir1 =
      face_number / 2 == 1 ?
#if MEMORY_TYPE == 1
        get_face_shape_gradients<Number>(mf_object_id) + shift :
#else
        shape_gradients + n_q_points_2d + shift :
#endif
        shape_gradients; // + offset1;

    Number *shape_gradient_dir2 =
      face_number / 2 == 2 ?
#if MEMORY_TYPE == 1
        get_face_shape_gradients<Number>(mf_object_id) + shift :
#else
        shape_gradients + n_q_points_2d + shift :
#endif
        shape_gradients; // + offset2;

    switch (dim)
      {
        case 1:
          {
            gradients<0, true, false, false>(shape_gradient_dir0, u, grad_u[0]);

            break;
          }
        case 2:
          {
            gradients<0, true, false, false>(shape_gradient_dir0, u, grad_u[0]);
            values<0, true, false, false>(shape_value_dir0, u, grad_u[1]);

            __syncthreads();

            values<1, true, false, true>(shape_value_dir1,
                                         grad_u[0],
                                         grad_u[0]);
            gradients<1, true, false, true>(shape_gradient_dir1,
                                            grad_u[1],
                                            grad_u[1]);

            break;
          }
        case 3:
          {
            gradients<0, true, false, false>(shape_gradient_dir0, u, grad_u[0]);
            values<0, true, false, false>(shape_value_dir0, u, grad_u[1]);
            values<0, true, false, false>(shape_value_dir0, u, grad_u[2]);

            __syncthreads();

            values<1, true, false, true>(shape_value_dir1,
                                         grad_u[0],
                                         grad_u[0]);
            gradients<1, true, false, true>(shape_gradient_dir1,
                                            grad_u[1],
                                            grad_u[1]);
            values<1, true, false, true>(shape_value_dir1,
                                         grad_u[2],
                                         grad_u[2]);

            __syncthreads();

            values<2, true, false, true>(shape_value_dir2,
                                         grad_u[0],
                                         grad_u[0]);
            values<2, true, false, true>(shape_value_dir2,
                                         grad_u[1],
                                         grad_u[1]);
            gradients<2, true, false, true>(shape_gradient_dir2,
                                            grad_u[2],
                                            grad_u[2]);

            break;
          }
        default:
          {
            // Do nothing. We should throw but we can't from a __device__
            // function.
            printf(
              "Error: Invalid dimension.\n In file cuda_tensor_product_kernels.cuh: "
              "EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::"
              "gradient_at_quad_pts().\n");
          }
      }
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  inline __device__ void
  EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::
    value_and_gradient_at_quad_pts(Number *const u, Number *grad_u[dim])
  {
    constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;

    const unsigned int shift = (face_number & 1) * n_q_points_2d;
    const unsigned int offset0 =
      compute_subface_offset<dim, n_q_points_2d, 0>(face_number,
                                                    subface_number);
    const unsigned int offset1 =
      compute_subface_offset<dim, n_q_points_2d, 1>(face_number,
                                                    subface_number);
    const unsigned int offset2 =
      compute_subface_offset<dim, n_q_points_2d, 2>(face_number,
                                                    subface_number);


    Number *shape_value_dir0 =
      face_number / 2 == 0 ?
        get_face_shape_values<Number>(mf_object_id) + shift :
        get_cell_shape_values<Number>(mf_object_id) + offset0;

    Number *shape_value_dir1 =
      face_number / 2 == 1 ?
        get_face_shape_values<Number>(mf_object_id) + shift :
        get_cell_shape_values<Number>(mf_object_id) + offset1;

    Number *shape_value_dir2 =
      face_number / 2 == 2 ?
        get_face_shape_values<Number>(mf_object_id) + shift :
        get_cell_shape_values<Number>(mf_object_id) + offset2;

    Number *co_shape_gradient_dir0 =
      face_number / 2 == 0 ?
        get_face_co_shape_gradients<Number>(mf_object_id) + shift :
        get_cell_co_shape_gradients<Number>(mf_object_id) + offset0;

    Number *co_shape_gradient_dir1 =
      face_number / 2 == 1 ?
        get_face_co_shape_gradients<Number>(mf_object_id) + shift :
        get_cell_co_shape_gradients<Number>(mf_object_id) + offset1;

    Number *co_shape_gradient_dir2 =
      face_number / 2 == 2 ?
        get_face_co_shape_gradients<Number>(mf_object_id) + shift :
        get_cell_co_shape_gradients<Number>(mf_object_id) + offset2;

    switch (dim)
      {
        case 1:
          {
            values<0, true, false, true>(shape_value_dir0, u, u);
            __syncthreads();

            gradients<0, true, false, false>(co_shape_gradient_dir0,
                                             u,
                                             grad_u[0]);

            break;
          }
        case 2:
          {
            values<0, true, false, true>(shape_value_dir0, u, u);
            __syncthreads();
            values<1, true, false, true>(shape_value_dir1, u, u);
            __syncthreads();

            gradients<0, true, false, false>(co_shape_gradient_dir0,
                                             u,
                                             grad_u[0]);
            gradients<1, true, false, false>(co_shape_gradient_dir1,
                                             u,
                                             grad_u[1]);

            break;
          }
        case 3:
          {
            values<0, true, false, true>(shape_value_dir0, u, u);
            __syncthreads();
            values<1, true, false, true>(shape_value_dir1, u, u);
            __syncthreads();
            values<2, true, false, true>(shape_value_dir2, u, u);
            __syncthreads();

            gradients<0, true, false, false>(co_shape_gradient_dir0,
                                             u,
                                             grad_u[0]);
            gradients<1, true, false, false>(co_shape_gradient_dir1,
                                             u,
                                             grad_u[1]);
            gradients<2, true, false, false>(co_shape_gradient_dir2,
                                             u,
                                             grad_u[2]);

            break;
          }
        default:
          {
            // Do nothing. We should throw but we can't from a __device__
            // function.
            printf(
              "Error: Invalid dimension.\n In file cuda_tensor_product_kernels.cuh: "
              "EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::"
              "value_and_gradient_at_quad_pts().\n");
          }
      }
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  template <bool add>
  inline __device__ void
  EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::
    integrate_gradient(Number *u, Number *grad_u[dim])
  {
    constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;

    const unsigned int shift = (face_number & 1) * n_q_points_2d;

    // // local refinement
    // const unsigned int offset0 =
    //   compute_subface_offset<dim, n_q_points_2d, 0>(face_number,
    //                                                 subface_number);
    // const unsigned int offset1 =
    //   compute_subface_offset<dim, n_q_points_2d, 1>(face_number,
    //                                                 subface_number);
    // const unsigned int offset2 =
    //   compute_subface_offset<dim, n_q_points_2d, 2>(face_number,
    //                                                 subface_number);


    Number *shape_value_dir0 =
      face_number / 2 == 0 ?
#if MEMORY_TYPE == 1
        get_face_shape_values<Number>(mf_object_id) + shift :
#else
        shape_values + n_q_points_2d + shift :
#endif
        shape_values; // + offset0;

    Number *shape_value_dir1 =
      face_number / 2 == 1 ?
#if MEMORY_TYPE == 1
        get_face_shape_values<Number>(mf_object_id) + shift :
#else
        shape_values + n_q_points_2d + shift :
#endif
        shape_values; // + offset1;

    Number *shape_value_dir2 =
      face_number / 2 == 2 ?
#if MEMORY_TYPE == 1
        get_face_shape_values<Number>(mf_object_id) + shift :
#else
        shape_values + n_q_points_2d + shift :
#endif
        shape_values; // + offset2;

    Number *shape_gradient_dir0 =
      face_number / 2 == 0 ?
#if MEMORY_TYPE == 1
        get_face_shape_gradients<Number>(mf_object_id) + shift :
#else
        shape_gradients + n_q_points_2d + shift :
#endif
        shape_gradients; // + offset0;

    Number *shape_gradient_dir1 =
      face_number / 2 == 1 ?
#if MEMORY_TYPE == 1
        get_face_shape_gradients<Number>(mf_object_id) + shift :
#else
        shape_gradients + n_q_points_2d + shift :
#endif
        shape_gradients; // + offset1;

    Number *shape_gradient_dir2 =
      face_number / 2 == 2 ?
#if MEMORY_TYPE == 1
        get_face_shape_gradients<Number>(mf_object_id) + shift :
#else
        shape_gradients + n_q_points_2d + shift :
#endif
        shape_gradients; // + offset2;

    switch (dim)
      {
        case 1:
          {
            gradients<0, false, add, false>(shape_gradient_dir0,
                                            grad_u[dim],
                                            u);

            break;
          }
        case 2:
          {
            gradients<0, false, false, true>(shape_gradient_dir0,
                                             grad_u[0],
                                             grad_u[0]);
            values<0, false, false, true>(shape_value_dir0,
                                          grad_u[1],
                                          grad_u[1]);

            __syncthreads();

            values<1, false, add, false>(shape_value_dir1, grad_u[0], u);
            __syncthreads();
            gradients<1, false, true, false>(shape_gradient_dir1, grad_u[1], u);

            break;
          }
        case 3:
          {
            gradients<0, false, false, true>(shape_gradient_dir0,
                                             grad_u[0],
                                             grad_u[0]);
            values<0, false, false, true>(shape_value_dir0,
                                          grad_u[1],
                                          grad_u[1]);
            values<0, false, false, true>(shape_value_dir0,
                                          grad_u[2],
                                          grad_u[2]);

            __syncthreads();

            values<1, false, false, true>(shape_value_dir1,
                                          grad_u[0],
                                          grad_u[0]);
            gradients<1, false, false, true>(shape_gradient_dir1,
                                             grad_u[1],
                                             grad_u[1]);
            values<1, false, false, true>(shape_value_dir1,
                                          grad_u[2],
                                          grad_u[2]);

            __syncthreads();

            values<2, false, add, false>(shape_value_dir2, grad_u[0], u);
            __syncthreads();
            values<2, false, true, false>(shape_value_dir2, grad_u[1], u);
            __syncthreads();
            gradients<2, false, true, false>(shape_gradient_dir2, grad_u[2], u);

            break;
          }
        default:
          {
            // Do nothing. We should throw but we can't from a __device__
            // function.
            printf(
              "Error: Invalid dimension.\n In file cuda_tensor_product_kernels.cuh: "
              "EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::"
              "integrate_gradient().\n");
          }
      }
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  inline __device__ void
  EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::
    integrate_value_and_gradient(Number *u, Number *grad_u[dim])
  {
    constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;

    const unsigned int shift = (face_number & 1) * n_q_points_2d;
    const unsigned int offset0 =
      compute_subface_offset<dim, n_q_points_2d, 0>(face_number,
                                                    subface_number);
    const unsigned int offset1 =
      compute_subface_offset<dim, n_q_points_2d, 1>(face_number,
                                                    subface_number);
    const unsigned int offset2 =
      compute_subface_offset<dim, n_q_points_2d, 2>(face_number,
                                                    subface_number);


    Number *shape_value_dir0 =
      face_number / 2 == 0 ?
        get_face_shape_values<Number>(mf_object_id) + shift :
        get_cell_shape_values<Number>(mf_object_id) + offset0;

    Number *shape_value_dir1 =
      face_number / 2 == 1 ?
        get_face_shape_values<Number>(mf_object_id) + shift :
        get_cell_shape_values<Number>(mf_object_id) + offset1;

    Number *shape_value_dir2 =
      face_number / 2 == 2 ?
        get_face_shape_values<Number>(mf_object_id) + shift :
        get_cell_shape_values<Number>(mf_object_id) + offset2;

    Number *co_shape_gradient_dir0 =
      face_number / 2 == 0 ?
        get_face_co_shape_gradients<Number>(mf_object_id) + shift :
        get_cell_co_shape_gradients<Number>(mf_object_id) + offset0;

    Number *co_shape_gradient_dir1 =
      face_number / 2 == 1 ?
        get_face_co_shape_gradients<Number>(mf_object_id) + shift :
        get_cell_co_shape_gradients<Number>(mf_object_id) + offset1;

    Number *co_shape_gradient_dir2 =
      face_number / 2 == 2 ?
        get_face_co_shape_gradients<Number>(mf_object_id) + shift :
        get_cell_co_shape_gradients<Number>(mf_object_id) + offset2;

    switch (dim)
      {
        case 1:
          {
            gradients<0, false, true, false>(co_shape_gradient_dir0,
                                             grad_u[0],
                                             u);
            __syncthreads();

            values<0, false, false, true>(shape_value_dir0, u, u);

            break;
          }
        case 2:
          {
            gradients<1, false, true, false>(co_shape_gradient_dir1,
                                             grad_u[1],
                                             u);
            __syncthreads();
            gradients<0, false, true, false>(co_shape_gradient_dir0,
                                             grad_u[0],
                                             u);
            __syncthreads();

            values<1, false, false, true>(shape_value_dir1, u, u);
            __syncthreads();
            values<0, false, false, true>(shape_value_dir0, u, u);
            __syncthreads();

            break;
          }
        case 3:
          {
            gradients<2, false, true, false>(co_shape_gradient_dir2,
                                             grad_u[2],
                                             u);
            __syncthreads();
            gradients<1, false, true, false>(co_shape_gradient_dir1,
                                             grad_u[1],
                                             u);
            __syncthreads();
            gradients<0, false, true, false>(co_shape_gradient_dir0,
                                             grad_u[0],
                                             u);
            __syncthreads();

            values<2, false, false, true>(shape_value_dir2, u, u);
            __syncthreads();
            values<1, false, false, true>(shape_value_dir1, u, u);
            __syncthreads();
            values<0, false, false, true>(shape_value_dir0, u, u);
            __syncthreads();

            break;
          }
        default:
          {
            // Do nothing. We should throw but we can't from a __device__
            // function.
            printf(
              "Error: Invalid dimension.\n In file cuda_tensor_product_kernels.cuh: "
              "EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::"
              "integrate_value_and_gradient().\n");
          }
      }
  }

} // namespace PSMF

#endif // CUDA_TENSOR_PRODUCT_KERNELS_CUH
