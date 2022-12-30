/**
 * @file cuda_patch_base.template.cuh
 * Created by Cu Cui on 2022/3/28.
 */

#include "loop_kernel.cuh"

namespace PSMF
{

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::Q>::
    LevelVertexPatch()
  {}

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::Q>::
    ~LevelVertexPatch()
  {
    free();
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::Q>::free()
  {
    for (auto &first_dof_color_ptr : first_dof)
      Utilities::CUDA::free(first_dof_color_ptr);
    first_dof.clear();

    for (auto &patch_id_color_ptr : patch_id)
      Utilities::CUDA::free(patch_id_color_ptr);
    patch_id.clear();

    Utilities::CUDA::free(eigenvalues);
    Utilities::CUDA::free(eigenvectors);
    Utilities::CUDA::free(global_mass_1d);
    Utilities::CUDA::free(global_derivative_1d);
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  std::size_t
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::Q>::
    memory_consumption() const
  {
    const unsigned int n_dofs_1d = 2 * fe_degree + 1;

    std::size_t result = 0;

    // For each color, add first_dof, patch_id, {mass,derivative}_matrix,
    // and eigen{values,vectors}.
    for (unsigned int i = 0; i < n_colors; ++i)
      {
        result += 2 * n_patches[i] * sizeof(unsigned int) +
                  2 * n_dofs_1d * n_dofs_1d * sizeof(Number) +
                  2 * n_dofs_1d * dim * sizeof(Number);
      }
    return result;
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  template <typename MatrixFreeType>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::Q>::reinit(
    const MatrixFreeType &matrix_free,
    const AdditionalData &additional_data)
  {
    if (typeid(Number) == typeid(double))
      cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    this->relaxation         = additional_data.relaxation;
    this->granularity_scheme = additional_data.granularity_scheme;

    dof_handler = &matrix_free->get_dof_handler();
    level       = matrix_free->get_mg_level();

    if (kernel == SmootherVariant::SEPERATE ||
        kernel == SmootherVariant::GLOBAL)
      matrix_free->initialize_dof_vector(tmp);

    switch (granularity_scheme)
      {
        case GranularityScheme::none:
          patch_per_block = 1;
          break;
        case GranularityScheme::user_define:
          patch_per_block = additional_data.patch_per_block;
          break;
        case GranularityScheme::multiple:
          patch_per_block = granularity_shmem<dim, fe_degree>();
          break;
        default:
          AssertThrow(false, ExcMessage("Invalid granularity scheme."));
          break;
      }

    rb_coloring();

    setup_color_arrays(n_colors);

    for (unsigned int i = 0; i < n_colors; ++i)
      {
        n_patches[i] = graph[i].size();
        setup_patch_arrays(i);
      }


    constexpr unsigned n_dofs_1d = 2 * fe_degree + 1;
    constexpr unsigned n_dofs_2d = n_dofs_1d * n_dofs_1d;

    alloc_arrays(&eigenvalues, n_dofs_1d);
    alloc_arrays(&eigenvectors, n_dofs_2d);
    alloc_arrays(&global_mass_1d, n_dofs_2d);
    alloc_arrays(&global_derivative_1d, n_dofs_2d);
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::Q>::Data
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::Q>::get_data(
    unsigned int color) const
  {
    Data data_copy;

    data_copy.n_patches            = n_patches[color];
    data_copy.patch_per_block      = patch_per_block;
    data_copy.relaxation           = relaxation;
    data_copy.first_dof            = first_dof[color];
    data_copy.patch_id             = patch_id[color];
    data_copy.eigenvalues          = eigenvalues;
    data_copy.eigenvectors         = eigenvectors;
    data_copy.global_mass_1d       = global_mass_1d;
    data_copy.global_derivative_1d = global_derivative_1d;

    return data_copy;
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  template <typename Functor, typename VectorType, typename Functor_inv>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::Q>::patch_loop(
    const Functor     &func,
    const VectorType  &src,
    VectorType        &dst,
    const Functor_inv &func_inv) const
  {
    switch (kernel)
      {
        case SmootherVariant::FUSED:
          patch_loop_fused(func, src, dst);
          break;
        case SmootherVariant::SEPERATE:
          patch_loop_seperate(func, func_inv, src, dst);
          break;
        default:
          AssertThrow(false, ExcMessage("Invalid Smoother Variant."));
          break;
      }
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  template <typename Functor, typename VectorType>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::Q>::
    patch_loop_fused(const Functor    &func,
                     const VectorType &src,
                     VectorType       &dst) const
  {
    auto shared_mem = [&]() {
      std::size_t mem = 0;

      const unsigned int n_dofs_1d = 2 * fe_degree + 1;
      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst, local_residual
      mem += 2 * patch_per_block * local_dim * sizeof(Number);
      // local_mass, local_derivative, local_eigenvectors, local_eigenvalues
      mem += 2 * 1 * n_dofs_1d * n_dofs_1d * 1 * sizeof(Number);
      // temp
      mem += (dim - 1) * patch_per_block * local_dim * sizeof(Number);

      return mem;
    };


    AssertCuda(cudaFuncSetAttribute(
      loop_kernel_fused<dim, fe_degree, Number, kernel, Functor, DoFLayout::Q>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_mem()));


    // loop over all patches
    switch (kernel)
      {
        case SmootherVariant::FUSED:
          for (unsigned int i = 0; i < n_colors; ++i)
            if (n_patches[i] > 0)
              loop_kernel_fused<dim,
                                fe_degree,
                                Number,
                                kernel,
                                Functor,
                                DoFLayout::Q>
                <<<grid_dim[i], block_dim[i], shared_mem()>>>(func,
                                                              src.get_values(),
                                                              dst.get_values(),
                                                              get_data(i));
          break;
        default:
          AssertThrow(false, ExcMessage("Invalid Smoother Variant."));
      }

    AssertCudaKernel();
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  template <typename Functor, typename Functor_inv, typename VectorType>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::Q>::
    patch_loop_seperate(const Functor     &func,
                        const Functor_inv &func_inv,
                        const VectorType  &src,
                        VectorType        &dst) const
  {
    auto shared_mem = [&]() {
      std::size_t mem = 0;

      const unsigned int n_dofs_1d = 2 * fe_degree + 1;
      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst, local_residual
      mem += 2 * patch_per_block * local_dim * sizeof(Number);
      // local_mass, local_derivative
      mem += 2 * 1 * n_dofs_1d * n_dofs_1d * 1 * sizeof(Number);
      // temp
      mem += (dim - 1) * patch_per_block * local_dim * sizeof(Number);

      return mem;
    };

    auto shared_mem_inv = [&]() {
      std::size_t mem = 0;

      const unsigned int n_dofs_1d = 2 * fe_degree - 1;
      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst, local_residual
      mem += 2 * patch_per_block * local_dim * sizeof(Number);
      // local_eigenvectors, local_eigenvalues
      mem += 2 * 1 * n_dofs_1d * n_dofs_1d * 1 * sizeof(Number);
      // temp
      mem += patch_per_block * local_dim * sizeof(Number);

      return mem;
    };


    AssertCuda(cudaFuncSetAttribute(loop_kernel_seperate<dim,
                                                         fe_degree,
                                                         Number,
                                                         kernel,
                                                         Functor,
                                                         DoFLayout::Q>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    shared_mem()));

    AssertCuda(cudaFuncSetAttribute(loop_kernel_seperate_inv<dim,
                                                             fe_degree,
                                                             Number,
                                                             kernel,
                                                             Functor_inv,
                                                             DoFLayout::Q>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    shared_mem_inv()));

    // loop over all patches
    for (unsigned int i = 0; i < n_colors; ++i)
      if (n_patches[i] > 0)
        {
          loop_kernel_seperate<dim,
                               fe_degree,
                               Number,
                               kernel,
                               Functor,
                               DoFLayout::Q>
            <<<grid_dim[i], block_dim[i], shared_mem()>>>(func,
                                                          src.get_values(),
                                                          dst.get_values(),
                                                          tmp.get_values(),
                                                          get_data(i));
          AssertCudaKernel();

          loop_kernel_seperate_inv<dim,
                                   fe_degree,
                                   Number,
                                   kernel,
                                   Functor_inv,
                                   DoFLayout::Q>
            <<<grid_dim[i], block_dim_inv[i], shared_mem_inv()>>>(
              func_inv, tmp.get_values(), dst.get_values(), get_data(i));
          AssertCudaKernel();
        }
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  template <typename MatrixType, typename Functor_inv, typename VectorType>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::Q>::
    patch_loop_global(const MatrixType  &A,
                      const Functor_inv &func_inv,
                      const VectorType  &src,
                      VectorType        &dst) const
  {
    auto shared_mem_inv = [&]() {
      std::size_t mem = 0;

      const unsigned int n_dofs_1d = 2 * fe_degree - 1;
      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst, local_residual
      mem += 2 * patch_per_block * local_dim * sizeof(Number);
      // local_eigenvectors, local_eigenvalues
      mem += 2 * 1 * n_dofs_1d * n_dofs_1d * 1 * sizeof(Number);
      // temp
      mem += (dim - 1) * patch_per_block * local_dim * sizeof(Number);

      return mem;
    };

    AssertCuda(cudaFuncSetAttribute(loop_kernel_seperate_inv<dim,
                                                             fe_degree,
                                                             Number,
                                                             kernel,
                                                             Functor_inv,
                                                             DoFLayout::Q>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    shared_mem_inv()));

    // loop over all patches
    for (unsigned int i = 0; i < n_colors; ++i)
      if (n_patches[i] > 0)
        {
          A->vmult(tmp, dst);
          tmp.sadd(-1., src);

          loop_kernel_seperate_inv<dim,
                                   fe_degree,
                                   Number,
                                   kernel,
                                   Functor_inv,
                                   DoFLayout::Q>
            <<<grid_dim[i], block_dim_inv[i], shared_mem_inv()>>>(
              func_inv, tmp.get_values(), dst.get_values(), get_data(i));
          AssertCudaKernel();
        }
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::Q>::
    reinit_tensor_product() const
  {
    FE_DGQ<1> fe_1d(fe_degree);

    constexpr unsigned int N    = fe_degree + 1;
    const Number scaling_factor = dim == 2 ? 1 : 1. / Util::pow(2, level);

    QGauss<1> quadrature(N);

    std::array<Table<2, Number>, dim> patch_mass;

    for (unsigned int d = 0; d < dim; ++d)
      {
        patch_mass[d].reinit(2 * N - 1, 2 * N - 1);
      }

    auto get_cell_laplace = [&]() {
      FullMatrix<double> cell_laplace(N, N);

      for (unsigned int i = 0; i < N; ++i)
        for (unsigned int j = 0; j < N; ++j)
          {
            double sum_laplace = 0;
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              {
                sum_laplace += (fe_1d.shape_grad(i, quadrature.point(q))[0] *
                                fe_1d.shape_grad(j, quadrature.point(q))[0]) *
                               quadrature.weight(q);
              }

            // scaling to real cells
            cell_laplace(i, j) = sum_laplace * scaling_factor;
          }

      return cell_laplace;
    };

    for (unsigned int i = 0; i < N; ++i)
      for (unsigned int j = 0; j < N; ++j)
        {
          double sum_mass = 0;
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            {
              sum_mass += (fe_1d.shape_value(i, quadrature.point(q)) *
                           fe_1d.shape_value(j, quadrature.point(q))) *
                          quadrature.weight(q);
            }
          for (unsigned int d = 0; d < dim; ++d)
            {
              patch_mass[d](i, j) += sum_mass;
              patch_mass[d](i + N - 1, j + N - 1) += sum_mass;
            }
        }

    auto laplace_middle = get_cell_laplace();

    // mass, laplace
    auto get_patch_laplace = [&](auto left, auto right) {
      std::array<Table<2, Number>, dim> patch_laplace;

      for (unsigned int d = 0; d < dim; ++d)
        {
          patch_laplace[d].reinit(2 * N - 1, 2 * N - 1);
        }

      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int i = 0; i < N; ++i)
          for (unsigned int j = 0; j < N; ++j)
            {
              patch_laplace[d](i, j) += left(i, j);
              patch_laplace[d](i + N - 1, j + N - 1) += right(i, j);
            }

      return patch_laplace;
    };

    auto patch_laplace = get_patch_laplace(laplace_middle, laplace_middle);

    // eigenvalue, eigenvector
    TensorProductData<dim, fe_degree, Number> tensor_product;
    if (level == 0)
      tensor_product.generate_1d_mf(1);
    else
      tensor_product.generate_1d_mf(level);
    std::array<std::array<unsigned int, 2>, dim> cell_indices;
    for (int d = 0; d < dim; ++d)
      {
        cell_indices[d][0] = 0;
        cell_indices[d][1] = 1;
      }
    tensor_product.template compute_tensor_product<true>(cell_indices);
    std::array<AlignedVector<Number>, dim> eigenval;
    std::array<Table<2, Number>, dim>      eigenvec;
    tensor_product.get_eigenvalues(eigenval);
    tensor_product.get_eigenvectors(eigenvec);

    constexpr unsigned int n_dofs_1d = 2 * fe_degree + 1;

    auto *mass    = new Number[n_dofs_1d * n_dofs_1d * dim];
    auto *laplace = new Number[n_dofs_1d * n_dofs_1d * dim];
    auto *values  = new Number[n_dofs_1d * n_dofs_1d * dim];
    auto *vectors = new Number[n_dofs_1d * n_dofs_1d * dim];

    for (int d = 0; d < dim; ++d)
      {
        std::transform(patch_mass[d].begin(),
                       patch_mass[d].end(),
                       &mass[n_dofs_1d * n_dofs_1d * d],
                       [](const Number m) -> Number { return m; });

        std::transform(patch_laplace[d].begin(),
                       patch_laplace[d].end(),
                       &laplace[n_dofs_1d * n_dofs_1d * d],
                       [](const Number m) -> Number { return m; });

        std::transform(eigenval[d].begin(),
                       eigenval[d].end(),
                       &values[n_dofs_1d * n_dofs_1d * d],
                       [](const Number m) -> Number { return m; });

        std::transform(eigenvec[d].begin(),
                       eigenvec[d].end(),
                       &vectors[n_dofs_1d * n_dofs_1d * d],
                       [](const Number m) -> Number { return m; });
      }

    cudaError_t error_code = cudaMemcpy(eigenvalues,
                                        values,
                                        n_dofs_1d * sizeof(Number),
                                        cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    error_code = cudaMemcpy(eigenvectors,
                            vectors,
                            n_dofs_1d * n_dofs_1d * sizeof(Number),
                            cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    error_code = cudaMemcpy(global_mass_1d,
                            mass,
                            n_dofs_1d * n_dofs_1d * sizeof(Number),
                            cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    error_code = cudaMemcpy(global_derivative_1d,
                            laplace,
                            n_dofs_1d * n_dofs_1d * sizeof(Number),
                            cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    delete[] mass;
    delete[] laplace;
    delete[] values;
    delete[] vectors;
  }


  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::Q>::rb_coloring()
  {
    n_colors = 1 << dim;
    graph.clear();
    graph.resize(n_colors);

    const unsigned int ncells_per_dim   = 1 << level;
    const unsigned int npatches_per_dim = ncells_per_dim - 1;

    const unsigned int gap    = 1 - 1;
    const unsigned int stride = 1 + 1 + gap;
    const unsigned int offset = 1;

    // left
    unsigned int       l = 2 * (1 + 1) - 2;
    const unsigned int left =
      std::min(ncells_per_dim % l + gap, (ncells_per_dim - 1) % l);

    if (dim == 2)
      {
        auto comp_color = [&, this](unsigned int jj,
                                    unsigned int ii,
                                    unsigned int color) {
          for (unsigned int j = jj; j < npatches_per_dim - left; j += stride)
            for (unsigned int i = ii; i < npatches_per_dim - left; i += stride)
              {
                for (unsigned int m = 0; m < offset; ++m)
                  for (unsigned int n = 0; n < offset; ++n)
                    graph[color].push_back((j + m) * npatches_per_dim +
                                           (i + n));
              }
        };
        unsigned int color = 0;
        for (unsigned int j = 0; j < 2; ++j)
          for (unsigned int i = 0; i < 2; ++i)
            {
              comp_color(j * offset, i * offset, color++);
            }
      }
    else if (dim == 3)
      {
        auto comp_color = [&, this](unsigned int kk,
                                    unsigned int jj,
                                    unsigned int ii,
                                    unsigned int color) {
          for (unsigned int k = kk; k < npatches_per_dim - left; k += stride)
            for (unsigned int j = jj; j < npatches_per_dim - left; j += stride)
              for (unsigned int i = ii; i < npatches_per_dim - left;
                   i += stride)
                {
                  for (unsigned int l = 0; l < offset; ++l)
                    for (unsigned int m = 0; m < offset; ++m)
                      for (unsigned int n = 0; n < offset; ++n)
                        graph[color].push_back(
                          (k + l) * npatches_per_dim * npatches_per_dim +
                          (j + m) * npatches_per_dim + i + n);
                }
        };
        unsigned int color = 0;
        for (unsigned int k = 0; k < 2; ++k)
          for (unsigned int j = 0; j < 2; ++j)
            for (unsigned int i = 0; i < 2; ++i)
              {
                comp_color(k * offset, j * offset, i * offset, color++);
              }
      }
    else
      Assert(false, ExcNotImplemented());

#ifdef DEBUG
    unsigned int sum = 0;
    for (auto &g : graph)
      sum += g.size();
    AssertDimension(sum, Util::pow(npatches_per_dim, dim));
#endif
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::Q>::
    setup_color_arrays(const unsigned int n_colors)
  {
    this->n_patches.resize(n_colors);
    this->grid_dim.resize(n_colors);
    this->block_dim.resize(n_colors);
    this->block_dim_inv.resize(n_colors);
    this->first_dof.resize(n_colors);
    this->patch_id.resize(n_colors);
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::Q>::
    setup_patch_arrays(const unsigned int color)
  {
    const unsigned int n_patch = n_patches[color];

    // Setup kernel parameters
    const double apply_n_blocks = std::ceil(
      static_cast<double>(n_patch) / static_cast<double>(patch_per_block));
    const unsigned int apply_x_n_blocks = std::round(std::sqrt(apply_n_blocks));
    const unsigned int apply_y_n_blocks =
      std::ceil(apply_n_blocks / static_cast<double>(apply_x_n_blocks));

    grid_dim[color] = dim3(apply_n_blocks);

    constexpr unsigned int n_dofs_1d     = 2 * fe_degree + 1;
    constexpr unsigned int n_dofs_1d_inv = 2 * fe_degree - 1;

    switch (kernel)
      {
        case SmootherVariant::FUSED:
          block_dim[color] = dim3(patch_per_block * n_dofs_1d, n_dofs_1d);
          break;
        case SmootherVariant::SEPERATE:
          block_dim[color] = dim3(patch_per_block * n_dofs_1d, n_dofs_1d);
          block_dim_inv[color] =
            dim3(patch_per_block * n_dofs_1d_inv, n_dofs_1d_inv);
          break;
        case SmootherVariant::GLOBAL:
          block_dim_inv[color] =
            dim3(patch_per_block * n_dofs_1d_inv, n_dofs_1d_inv);
          break;
        default:
          AssertThrow(false, ExcMessage("Invalid Smoother Variant."));
          break;
      }

    const unsigned int n_patch_1d    = (1 << level) - 1;
    auto               get_first_dof = [&](unsigned int patch_id) {
      const unsigned int n_cells = n_patch_1d + 1;
      unsigned int       cells[dim];
      unsigned int       first_dof;
      if (dim == 2)
        {
          cells[0] = patch_id % n_patch_1d;
          cells[1] = patch_id / n_patch_1d;

          first_dof =
            cells[0] * fe_degree +
            cells[1] * fe_degree * Util::pow(n_cells * fe_degree + 1, 1);
        }
      else if (dim == 3)
        {
          cells[0] = patch_id % n_patch_1d;
          cells[1] = (patch_id / n_patch_1d) % n_patch_1d;
          cells[2] = patch_id / (n_patch_1d * n_patch_1d);

          first_dof =
            cells[0] * fe_degree +
            cells[1] * fe_degree * Util::pow(n_cells * fe_degree + 1, 1) +
            cells[2] * fe_degree * Util::pow(n_cells * fe_degree + 1, 2);
        }
      return first_dof;
    };

    // Alloc arrays on device
    alloc_arrays(&first_dof[color], n_patch);
    alloc_arrays(&patch_id[color], n_patch);

    auto *first_dof_host = new unsigned int[n_patch];

    for (unsigned int p = 0; p < n_patch; ++p)
      first_dof_host[p] = get_first_dof(graph[color][p]);

    cudaError_t error_code = cudaMemcpy(first_dof[color],
                                        first_dof_host,
                                        n_patch * sizeof(unsigned int),
                                        cudaMemcpyHostToDevice);
    AssertCuda(error_code);
    error_code = cudaMemcpy(patch_id[color],
                            graph[color].data(),
                            n_patch * sizeof(unsigned int),
                            cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    delete[] first_dof_host;
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  template <typename Number1>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::Q>::alloc_arrays(
    Number1          **array_device,
    const unsigned int n)
  {
    cudaError_t error_code = cudaMalloc(array_device, n * sizeof(Number1));
    AssertCuda(error_code);
  }

} // namespace PSMF

/**
 * \page patch_base.template
 * \include patch_base.template.cuh
 */