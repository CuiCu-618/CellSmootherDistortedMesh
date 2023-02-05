/**
 * @file patch_base.template.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief This class collects all the data that is stored for the matrix free implementation.
 * @version 1.0
 * @date 2023-02-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "loop_kernel.cuh"

namespace PSMF
{

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::DGQ>::
    LevelVertexPatch()
  {}

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::DGQ>::
    ~LevelVertexPatch()
  {
    free();
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::DGQ>::free()
  {
    for (auto &first_dof_color_ptr : first_dof)
      Utilities::CUDA::free(first_dof_color_ptr);
    first_dof.clear();

    for (auto &patch_id_color_ptr : patch_id)
      Utilities::CUDA::free(patch_id_color_ptr);
    patch_id.clear();

    for (auto &patch_type_color_ptr : patch_type)
      Utilities::CUDA::free(patch_type_color_ptr);
    patch_type.clear();

    // Utilities::CUDA::free(eigenvalues);
    // Utilities::CUDA::free(eigenvectors);
    // Utilities::CUDA::free(global_mass_1d);
    // Utilities::CUDA::free(global_derivative_1d);

    ordering_to_type.clear();
    patch_id_host.clear();
    patch_type_host.clear();
    first_dof_host.clear();
    h_to_l_host.clear();
    l_to_h_host.clear();
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  std::size_t
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::DGQ>::
    memory_consumption() const
  {
    const unsigned int n_dofs_1d = 2 * fe_degree + 2;

    std::size_t result = 0;

    // For each color, add first_dof, patch_id, {mass,derivative}_matrix,
    // and eigen{values,vectors}.
    for (unsigned int i = 0; i < n_colors; ++i)
      {
        result += 2 * n_patches[i] * sizeof(unsigned int) +
                  2 * n_dofs_1d * n_dofs_1d * (1 << level) * sizeof(Number) +
                  2 * n_dofs_1d * dim * sizeof(Number);
      }
    return result;
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  std::vector<std::vector<
    typename LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::DGQ>::
      CellIterator>>
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::DGQ>::
    gather_vertex_patches(const DoFHandler<dim> &dof_handler,
                          const unsigned int     level) const
  {
    // LAMBDA checks if a vertex is at the physical boundary
    auto &&is_boundary_vertex = [](const CellIterator &cell,
                                   const unsigned int  vertex_id) {
      return std::any_of(
        std::begin(GeometryInfo<dim>::vertex_to_face[vertex_id]),
        std::end(GeometryInfo<dim>::vertex_to_face[vertex_id]),
        [&cell](const auto &face_no) { return cell->at_boundary(face_no); });
    };

    const auto locally_owned_range_mg =
      filter_iterators(dof_handler.mg_cell_iterators_on_level(level),
                       IteratorFilters::LocallyOwnedLevelCell());
    /**
     * A mapping @p global_to_local_map between the global vertex and
     * the pair containing the number of locally owned cells and the
     * number of all cells (including ghosts) is constructed
     */
    std::map<unsigned int, std::pair<unsigned int, unsigned int>>
      global_to_local_map;
    for (const auto &cell : locally_owned_range_mg)
      {
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          if (!is_boundary_vertex(cell, v))
            {
              const unsigned int global_index = cell->vertex_index(v);
              const auto element = global_to_local_map.find(global_index);
              if (element != global_to_local_map.cend())
                {
                  ++(element->second.first);
                  ++(element->second.second);
                }
              else
                {
                  const auto n_cells_pair = std::pair<unsigned, unsigned>{1, 1};
                  const auto status       = global_to_local_map.insert(
                    std::make_pair(global_index, n_cells_pair));
                  (void)status;
                  Assert(status.second,
                         ExcMessage("failed to insert key-value-pair"))
                }
            }
      }

    /**
     * Enumerate the patches contained in @p global_to_local_map by
     * replacing the former number of locally owned cells in terms of a
     * consecutive numbering. The local numbering is required for
     * gathering the level cell iterators into a collection @
     * cell_collections according to the global vertex index.
     */
    unsigned int local_index = 0;
    for (auto &key_value : global_to_local_map)
      {
        key_value.second.first = local_index++;
      }
    const unsigned n_subdomains = global_to_local_map.size();
    AssertDimension(n_subdomains, local_index);
    std::vector<std::vector<CellIterator>> cell_collections;
    cell_collections.resize(n_subdomains);
    for (auto &cell : dof_handler.mg_cell_iterators_on_level(level))
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          const unsigned int global_index = cell->vertex_index(v);
          const auto         element = global_to_local_map.find(global_index);
          if (element != global_to_local_map.cend())
            {
              const unsigned int local_index = element->second.first;
              const unsigned int patch_size  = element->second.second;
              auto              &collection  = cell_collections[local_index];
              if (collection.empty())
                collection.resize(patch_size);
              if (patch_size == regular_vpatch_size) // regular patch
                collection[regular_vpatch_size - 1 - v] = cell;
              else // irregular patch
                AssertThrow(false, ExcMessage("TODO irregular vertex patches"));
            }
        }
    return cell_collections;
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::DGQ>::
    get_patch_data(const PatchIterator &patch, const unsigned int patch_id)
  {
    std::vector<unsigned int> local_dof_indices(Util::pow(fe_degree + 1, dim));
    std::vector<unsigned int> numbering(regular_vpatch_size);
    std::iota(numbering.begin(), numbering.end(), 0);

    // first_dof
    for (unsigned int cell = 0; cell < regular_vpatch_size; ++cell)
      {
        auto cell_ptr = (*patch)[cell];
        cell_ptr->get_mg_dof_indices(local_dof_indices);
        first_dof_host[patch_id * regular_vpatch_size + cell] =
          local_dof_indices[0];
      }

    // patch_type. TODO: Fix: only works on [0,1]^d
    // TODO: level == 1, one patch only.
    const double h            = 1. / Util::pow(2, level);
    auto         first_center = (*patch)[0]->center();

    if (level == 1)
      for (unsigned int d = 0; d < dim; ++d)
        patch_type_host[patch_id * dim + d] = 2;
    else
      for (unsigned int d = 0; d < dim; ++d)
        {
          auto pos = std::floor(first_center[d] / h + 1 / 3);
          patch_type_host[patch_id * dim + d] =
            (pos > 0) + (pos == (Util::pow(2, level) - 2));
        }


    // patch_id
    std::sort(numbering.begin(),
              numbering.end(),
              [&](unsigned lhs, unsigned rhs) {
                return first_dof_host[patch_id * regular_vpatch_size + lhs] <
                       first_dof_host[patch_id * regular_vpatch_size + rhs];
              });

    auto encode = [&](unsigned int sum, int val) { return sum * 10 + val; };
    unsigned int label =
      std::accumulate(numbering.begin(), numbering.end(), 0, encode);

    const auto element = ordering_to_type.find(label);
    if (element != ordering_to_type.end()) // Fouond
      {
        patch_id_host[patch_id] = element->second;
      }
    else // Not found
      {
        ordering_to_type.insert({label, ordering_types++});
        patch_id_host[patch_id] = ordering_to_type[label];
      }
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::DGQ>::reinit(
    const DoFHandler<dim> &mg_dof,
    const unsigned int     mg_level,
    const AdditionalData  &additional_data)
  {
    if (typeid(Number) == typeid(double))
      cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    this->relaxation         = additional_data.relaxation;
    this->use_coloring       = additional_data.use_coloring;
    this->granularity_scheme = additional_data.granularity_scheme;

    dof_handler = &mg_dof;
    level       = mg_level;

    if (kernel == SmootherVariant::SEPERATE ||
        kernel == SmootherVariant::GLOBAL)
      tmp.reinit(mg_dof.n_dofs());
    // matrix_free->initialize_dof_vector(tmp);

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

    // create patches
    std::vector<std::vector<CellIterator>> cell_collections;
    cell_collections = std::move(gather_vertex_patches(*dof_handler, level));

    if (use_coloring)
      {
        graph_ptr.clear();
        graph_ptr.resize(1 << dim);
        n_colors = graph_ptr.size();
        for (auto patch = cell_collections.begin();
             patch != cell_collections.end();
             ++patch)
          {
            auto first_cell = (*patch)[0];

            graph_ptr[first_cell->parent()->child_iterator_to_index(first_cell)]
              .push_back(patch);
          }
      }
    else
      {
        n_colors = 1;
        graph_ptr.clear();
        graph_ptr.resize(n_colors);

        for (auto patch = cell_collections.begin();
             patch != cell_collections.end();
             ++patch)
          graph_ptr[0].push_back(patch);
      }

    setup_color_arrays(n_colors);

    ordering_to_type.clear();
    ordering_types = 0;
    for (unsigned int i = 0; i < n_colors; ++i)
      {
        n_patches[i] = graph_ptr[i].size();
        patch_type_host.clear();
        patch_id_host.clear();
        first_dof_host.clear();
        patch_id_host.resize(n_patches[i]);
        patch_type_host.resize(n_patches[i] * dim);
        first_dof_host.resize(n_patches[i] * regular_vpatch_size);

        setup_patch_arrays(i);
        auto patch = graph_ptr[i].begin(), end_patch = graph_ptr[i].end();
        for (unsigned int p_id = 0; patch != end_patch; ++patch, ++p_id)
          get_patch_data(*patch, p_id);

        // alloc_and_copy_arrays(i);
        alloc_arrays(&first_dof[i], n_patches[i] * regular_vpatch_size);
        alloc_arrays(&patch_id[i], n_patches[i]);
        alloc_arrays(&patch_type[i], n_patches[i] * dim);

        cudaError_t error_code = cudaMemcpy(patch_id[i],
                                            patch_id_host.data(),
                                            n_patches[i] * sizeof(unsigned int),
                                            cudaMemcpyHostToDevice);
        AssertCuda(error_code);

        error_code =
          cudaMemcpy(first_dof[i],
                     first_dof_host.data(),
                     regular_vpatch_size * n_patches[i] * sizeof(unsigned int),
                     cudaMemcpyHostToDevice);
        AssertCuda(error_code);

        error_code = cudaMemcpy(patch_type[i],
                                patch_type_host.data(),
                                dim * n_patches[i] * sizeof(unsigned int),
                                cudaMemcpyHostToDevice);
        AssertCuda(error_code);
      }

    // Mapping
    if (dim == 2)
      {
        lookup_table.insert({123, {{0, 1}}}); // x-y
        lookup_table.insert({213, {{1, 0}}}); // y-x
      }
    else if (dim == 3)
      {
        lookup_table.insert({1234567, {{0, 1, 2}}}); // x-y-z
        lookup_table.insert({1452367, {{0, 2, 1}}}); // x-z-y
        lookup_table.insert({2134657, {{1, 0, 2}}}); // y-x-z
        lookup_table.insert({2461357, {{1, 2, 0}}}); // y-z-x
        lookup_table.insert({4152637, {{2, 0, 1}}}); // z-x-y
        lookup_table.insert({4261537, {{2, 1, 0}}}); // z-y-x
      }

    constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;
    constexpr unsigned int N         = fe_degree + 1;
    constexpr unsigned int z         = dim == 2 ? 1 : fe_degree + 1;
    h_to_l_host.resize(Util::pow(n_dofs_1d, dim) * dim * (dim - 1));
    l_to_h_host.resize(Util::pow(n_dofs_1d, dim) * dim * (dim - 1));

    auto generate_indices = [&](unsigned int label, unsigned int type) {
      const unsigned int          offset = type * Util::pow(n_dofs_1d, dim);
      std::array<unsigned int, 3> strides;
      for (unsigned int i = 0; i < 3; ++i)
        strides[i] = Util::pow(n_dofs_1d, lookup_table[label][i]);

      unsigned int count = 0;

      for (unsigned i = 0; i < dim - 1; ++i)
        for (unsigned int j = 0; j < 2; ++j)
          for (unsigned int k = 0; k < 2; ++k)
            for (unsigned int l = 0; l < z; ++l)
              for (unsigned int m = 0; m < fe_degree + 1; ++m)
                for (unsigned int n = 0; n < fe_degree + 1; ++n)
                  {
                    h_to_l_host[offset + (i * N) * strides[2] +
                                l * n_dofs_1d * n_dofs_1d +
                                (j * N) * strides[1] + m * n_dofs_1d +
                                (k * N) * strides[0] + n] = count;
                    l_to_h_host[offset + count++] =
                      (i * N) * strides[2] + l * n_dofs_1d * n_dofs_1d +
                      (j * N) * strides[1] + m * n_dofs_1d +
                      (k * N) * strides[0] + n;
                  }
    };
    for (auto &el : ordering_to_type)
      generate_indices(el.first, el.second);

    alloc_arrays(&l_to_h, Util::pow(n_dofs_1d, dim) * dim * (dim - 1));
    alloc_arrays(&h_to_l, Util::pow(n_dofs_1d, dim) * dim * (dim - 1));

    cudaError_t error_code = cudaMemcpy(l_to_h,
                                        l_to_h_host.data(),
                                        Util::pow(n_dofs_1d, dim) * dim *
                                          (dim - 1) * sizeof(unsigned int),
                                        cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    error_code = cudaMemcpy(h_to_l,
                            h_to_l_host.data(),
                            Util::pow(n_dofs_1d, dim) * dim * (dim - 1) *
                              sizeof(unsigned int),
                            cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    constexpr unsigned n_dofs_2d = n_dofs_1d * n_dofs_1d;

    alloc_arrays(&eigenvalues, n_dofs_1d);
    alloc_arrays(&eigenvectors, n_dofs_2d);
    alloc_arrays(&global_mass_1d, n_dofs_2d * 3);
    alloc_arrays(&global_derivative_1d, n_dofs_2d * 3);
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::DGQ>::Data
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::DGQ>::get_data(
    unsigned int color) const
  {
    Data data_copy;

    data_copy.n_patches            = n_patches[color];
    data_copy.patch_per_block      = patch_per_block;
    data_copy.relaxation           = relaxation;
    data_copy.first_dof            = first_dof[color];
    data_copy.patch_id             = patch_id[color];
    data_copy.patch_type           = patch_type[color];
    data_copy.l_to_h               = l_to_h;
    data_copy.h_to_l               = h_to_l;
    data_copy.eigenvalues          = eigenvalues;
    data_copy.eigenvectors         = eigenvectors;
    data_copy.global_mass_1d       = global_mass_1d;
    data_copy.global_derivative_1d = global_derivative_1d;

    return data_copy;
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  template <typename Functor, typename VectorType, typename Functor_inv>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::DGQ>::patch_loop(
    const Functor     &func,
    const VectorType  &src,
    VectorType        &dst,
    const Functor_inv &func_inv) const
  {
    switch (kernel)
      {
        case SmootherVariant::SEPERATE:
          patch_loop_seperate(func, func_inv, src, dst);
          break;
        case SmootherVariant::FUSED_BASE:
          patch_loop_fused(func, src, dst);
          break;
        case SmootherVariant::FUSED_L:
          patch_loop_fused(func, src, dst);
          break;
        case SmootherVariant::FUSED_3D:
          patch_loop_fused(func, src, dst);
          break;
        case SmootherVariant::FUSED_CF:
          patch_loop_fused(func, src, dst);
          break;
        default:
          AssertThrow(false, ExcMessage("Invalid Smoother Variant."));
          break;
      }
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  template <typename Functor, typename VectorType>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::DGQ>::
    patch_loop_fused(const Functor    &func,
                     const VectorType &src,
                     VectorType       &dst) const
  {
    auto shared_mem = [&]() {
      std::size_t mem = 0;

      const unsigned int n_dofs_1d = 2 * fe_degree + 2;
      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      mem += 1 * patch_per_block * local_dim * sizeof(unsigned int);
      // local_src, local_dst, local_residual
      mem += 2 * patch_per_block * local_dim * sizeof(Number);
      // local_mass, local_derivative, local_eigenvectors, local_eigenvalues
      mem += 2 * 1 * n_dofs_1d * n_dofs_1d * dim * sizeof(Number);
      // temp
      mem += (dim - 1) * patch_per_block * local_dim * sizeof(Number);

      return mem;
    };

    // loop over all patches
    switch (kernel)
      {
        case SmootherVariant::FUSED_BASE:
          AssertCuda(
            cudaFuncSetAttribute(loop_kernel_fused_base<dim,
                                                        fe_degree,
                                                        Number,
                                                        kernel,
                                                        Functor,
                                                        DoFLayout::DGQ>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem()));
          for (unsigned int i = 0; i < n_colors; ++i)
            if (n_patches[i] > 0)
              loop_kernel_fused_base<dim,
                                     fe_degree,
                                     Number,
                                     kernel,
                                     Functor,
                                     DoFLayout::DGQ>
                <<<grid_dim[i], block_dim[i], shared_mem()>>>(func,
                                                              src.get_values(),
                                                              dst.get_values(),
                                                              get_data(i));
          break;
        case SmootherVariant::FUSED_L:
          AssertCuda(
            cudaFuncSetAttribute(loop_kernel_fused_l<dim,
                                                     fe_degree,
                                                     Number,
                                                     kernel,
                                                     Functor,
                                                     DoFLayout::DGQ>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem()));
          for (unsigned int i = 0; i < n_colors; ++i)
            if (n_patches[i] > 0)
              loop_kernel_fused_l<dim,
                                  fe_degree,
                                  Number,
                                  kernel,
                                  Functor,
                                  DoFLayout::DGQ>
                <<<grid_dim[i], block_dim[i], shared_mem()>>>(func,
                                                              src.get_values(),
                                                              dst.get_values(),
                                                              get_data(i));
          break;
        case SmootherVariant::FUSED_3D:
          AssertCuda(
            cudaFuncSetAttribute(loop_kernel_fused_3d<dim,
                                                      fe_degree,
                                                      Number,
                                                      kernel,
                                                      Functor,
                                                      DoFLayout::DGQ>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem()));
          for (unsigned int i = 0; i < n_colors; ++i)
            if (n_patches[i] > 0)
              loop_kernel_fused_3d<dim,
                                   fe_degree,
                                   Number,
                                   kernel,
                                   Functor,
                                   DoFLayout::DGQ>
                <<<grid_dim[i], block_dim[i], shared_mem()>>>(func,
                                                              src.get_values(),
                                                              dst.get_values(),
                                                              get_data(i));
          break;
        case SmootherVariant::FUSED_CF:
          AssertCuda(
            cudaFuncSetAttribute(loop_kernel_fused_cf<dim,
                                                      fe_degree,
                                                      Number,
                                                      kernel,
                                                      Functor,
                                                      DoFLayout::DGQ>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem()));
          for (unsigned int i = 0; i < n_colors; ++i)
            if (n_patches[i] > 0)
              loop_kernel_fused_cf<dim,
                                   fe_degree,
                                   Number,
                                   kernel,
                                   Functor,
                                   DoFLayout::DGQ>
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
  template <typename Functor, typename VectorType>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::DGQ>::cell_loop(
    const Functor    &func,
    const VectorType &src,
    VectorType       &dst) const
  {
    auto shared_mem = [&]() {
      std::size_t mem = 0;

      const unsigned int n_dofs_1d = 2 * fe_degree + 2;
      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // global_dof_indices;
      // For global_dof_indices_ori and l_to_h, unsigned int is enough.
      mem += 1 * patch_per_block * local_dim * sizeof(unsigned int);
      // local_src, local_dst
      mem += 2 * patch_per_block * local_dim * sizeof(Number);
      // local_mass, local_derivative
      mem += 2 * patch_per_block * n_dofs_1d * n_dofs_1d * dim * sizeof(Number);
      // temp
      mem += (dim - 1) * patch_per_block * local_dim * sizeof(Number);

      return mem;
    };

    AssertCuda(cudaFuncSetAttribute(laplace_kernel_base<dim,
                                                        fe_degree,
                                                        Number,
                                                        kernel,
                                                        Functor,
                                                        DoFLayout::DGQ>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    shared_mem()));

    for (unsigned int i = 0; i < n_colors; ++i)
      if (n_patches[i] > 0)
        {
          laplace_kernel_base<dim,
                              fe_degree,
                              Number,
                              kernel,
                              Functor,
                              DoFLayout::DGQ>
            <<<grid_dim[i], block_dim[i], shared_mem()>>>(func,
                                                          src.get_values(),
                                                          dst.get_values(),
                                                          get_data(i));
          AssertCudaKernel();
        }
  }

  // TODO:
  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  template <typename Functor, typename Functor_inv, typename VectorType>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::DGQ>::
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
                                                         DoFLayout::DGQ>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    shared_mem()));

    AssertCuda(cudaFuncSetAttribute(loop_kernel_seperate_inv<dim,
                                                             fe_degree,
                                                             Number,
                                                             kernel,
                                                             Functor_inv,
                                                             DoFLayout::DGQ>,
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
                               DoFLayout::DGQ>
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
                                   DoFLayout::DGQ>
            <<<grid_dim[i], block_dim_inv[i], shared_mem_inv()>>>(
              func_inv, tmp.get_values(), dst.get_values(), get_data(i));
          AssertCudaKernel();
        }
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  template <typename MatrixType, typename Functor_inv, typename VectorType>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::DGQ>::
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
                                                             DoFLayout::DGQ>,
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
                                   DoFLayout::DGQ>
            <<<grid_dim[i], block_dim_inv[i], shared_mem_inv()>>>(
              func_inv, tmp.get_values(), dst.get_values(), get_data(i));
          AssertCudaKernel();
        }
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::DGQ>::
    reinit_tensor_product_smoother() const
  {
    std::string name = dof_handler->get_fe().get_name();
    name.replace(name.find('<') + 1, 1, "1");
    std::unique_ptr<FiniteElement<1>> fe_1d = FETools::get_fe_by_name<1>(name);

    constexpr unsigned int N              = fe_degree + 1;
    constexpr Number       penalty_factor = 1.0 * fe_degree * (fe_degree + 1);
    const Number scaling_factor = dim == 2 ? 1 : 1. / Util::pow(2, level);

    QGauss<1> quadrature(N);

    FullMatrix<double> laplace_interface_mixed(N, N);
    FullMatrix<double> laplace_interface_penalty(N, N);

    std::array<Table<2, Number>, dim> patch_mass;

    for (unsigned int d = 0; d < dim; ++d)
      {
        patch_mass[d].reinit(2 * N, 2 * N);
      }

    auto get_cell_laplace = [&](unsigned int type) {
      FullMatrix<double> cell_laplace(N, N);

      Number boundary_factor_left  = 1.;
      Number boundary_factor_right = 1.;

      if (type == 0)
        boundary_factor_left = 2.;
      else if (type == 1)
        {}
      else if (type == 2)
        boundary_factor_right = 2.;
      else
        Assert(false, ExcNotImplemented());

      for (unsigned int i = 0; i < N; ++i)
        for (unsigned int j = 0; j < N; ++j)
          {
            double sum_laplace = 0;
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              {
                sum_laplace += (fe_1d->shape_grad(i, quadrature.point(q))[0] *
                                fe_1d->shape_grad(j, quadrature.point(q))[0]) *
                               quadrature.weight(q);
              }

            sum_laplace +=
              boundary_factor_left *
              (1. * fe_1d->shape_value(i, Point<1>()) *
                 fe_1d->shape_value(j, Point<1>()) * penalty_factor +
               0.5 * fe_1d->shape_grad(i, Point<1>())[0] *
                 fe_1d->shape_value(j, Point<1>()) +
               0.5 * fe_1d->shape_grad(j, Point<1>())[0] *
                 fe_1d->shape_value(i, Point<1>()));

            sum_laplace +=
              boundary_factor_right *
              (1. * fe_1d->shape_value(i, Point<1>(1.0)) *
                 fe_1d->shape_value(j, Point<1>(1.0)) * penalty_factor -
               0.5 * fe_1d->shape_grad(i, Point<1>(1.0))[0] *
                 fe_1d->shape_value(j, Point<1>(1.0)) -
               0.5 * fe_1d->shape_grad(j, Point<1>(1.0))[0] *
                 fe_1d->shape_value(i, Point<1>(1.0)));

            // scaling to real cells
            cell_laplace(i, j) = sum_laplace * scaling_factor;
          }

      return cell_laplace;
    };

    for (unsigned int i = 0; i < N; ++i)
      for (unsigned int j = 0; j < N; ++j)
        {
          double sum_mass = 0, sum_mixed = 0, sum_penalty = 0;
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            {
              sum_mass += (fe_1d->shape_value(i, quadrature.point(q)) *
                           fe_1d->shape_value(j, quadrature.point(q))) *
                          quadrature.weight(q);
            }
          for (unsigned int d = 0; d < dim; ++d)
            {
              patch_mass[d](i, j)         = sum_mass;
              patch_mass[d](i + N, j + N) = sum_mass;
            }

          sum_mixed += (-0.5 * fe_1d->shape_grad(i, Point<1>())[0] *
                        fe_1d->shape_value(j, Point<1>(1.0)));

          sum_penalty +=
            (-1. * fe_1d->shape_value(i, Point<1>()) *
             fe_1d->shape_value(j, Point<1>(1.0)) * penalty_factor);

          laplace_interface_mixed(N - 1 - i, N - 1 - j) =
            scaling_factor * sum_mixed;
          laplace_interface_penalty(N - 1 - i, N - 1 - j) =
            scaling_factor * sum_penalty;
        }

    auto laplace_left   = get_cell_laplace(0);
    auto laplace_middle = get_cell_laplace(1);
    auto laplace_right  = get_cell_laplace(2);

    // mass, laplace
    auto get_patch_laplace = [&](auto left, auto right) {
      std::array<Table<2, Number>, dim> patch_laplace;

      for (unsigned int d = 0; d < dim; ++d)
        {
          patch_laplace[d].reinit(2 * N, 2 * N);
        }

      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int i = 0; i < N; ++i)
          for (unsigned int j = 0; j < N; ++j)
            {
              patch_laplace[d](i, j)         = left(i, j);
              patch_laplace[d](i + N, j + N) = right(i, j);

              patch_laplace[d](i, j + N) = laplace_interface_mixed(i, j);
              patch_laplace[d](i, j + N) +=
                laplace_interface_mixed(N - 1 - j, N - 1 - i);
              patch_laplace[d](i, j + N) +=
                laplace_interface_penalty(N - 1 - j, N - 1 - i);
              patch_laplace[d](j + N, i) = patch_laplace[d](i, j + N);
            }

      return patch_laplace;
    };

    auto patch_laplace = get_patch_laplace(laplace_middle, laplace_middle);

    // eigenvalue, eigenvector
    std::array<Table<2, Number>, dim> patch_mass_inv;
    std::array<Table<2, Number>, dim> patch_laplace_inv;

    for (unsigned int d = 0; d < dim; ++d)
      {
        patch_mass_inv[d].reinit(2 * N - 2, 2 * N - 2);
        patch_laplace_inv[d].reinit(2 * N - 2, 2 * N - 2);
      }

    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int i = 0; i < 2 * N - 2; ++i)
        for (unsigned int j = 0; j < 2 * N - 2; ++j)
          {
            patch_mass_inv[d](i, j)    = patch_mass[d](i + 1, j + 1);
            patch_laplace_inv[d](i, j) = patch_laplace[d](i + 1, j + 1);
          }


    // eigenvalue, eigenvector
    TensorProductData<dim, fe_degree, Number> tensor_product;
    tensor_product.reinit(patch_mass_inv, patch_laplace_inv);

    std::array<AlignedVector<Number>, dim> eigenval;
    std::array<Table<2, Number>, dim>      eigenvec;
    tensor_product.get_eigenvalues(eigenval);
    tensor_product.get_eigenvectors(eigenvec);

    constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;

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

    // for (unsigned int i = 0; i < n_dofs_1d * n_dofs_1d; ++i)
    //   std::cout << mass[i] << " " << laplace[i] << " " << vectors[i]
    //             << std::endl;

    delete[] mass;
    delete[] laplace;
    delete[] values;
    delete[] vectors;
  }


  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::DGQ>::
    reinit_tensor_product_laplace() const
  {
    std::string name = dof_handler->get_fe().get_name();
    name.replace(name.find('<') + 1, 1, "1");
    std::unique_ptr<FiniteElement<1>> fe_1d = FETools::get_fe_by_name<1>(name);

    constexpr unsigned int N              = fe_degree + 1;
    constexpr Number       penalty_factor = 1.0 * fe_degree * (fe_degree + 1);
    const Number scaling_factor = dim == 2 ? 1 : 1. / Util::pow(2, level);

    QGauss<1> quadrature(N);

    FullMatrix<double> laplace_interface_mixed(N, N);
    FullMatrix<double> laplace_interface_penalty(N, N);

    Table<2, Number> patch_mass_0;
    Table<2, Number> patch_mass_1;
    patch_mass_0.reinit(2 * N, 2 * N);
    patch_mass_1.reinit(2 * N, 2 * N);

    auto get_cell_laplace = [&](unsigned int type, unsigned int pos) {
      FullMatrix<double> cell_laplace(N, N);

      Number boundary_factor_left  = 1.;
      Number boundary_factor_right = 1.;

      unsigned int is_first = pos == 0 ? 1 : 0;

      if (type == 0)
        boundary_factor_left = 2.;
      else if (type == 1 && pos == 0)
        boundary_factor_left = 0.;
      else if (type == 1 && pos == 1)
        boundary_factor_right = 0.;
      else if (type == 2)
        boundary_factor_right = 2.;

      for (unsigned int i = 0; i < N; ++i)
        for (unsigned int j = 0; j < N; ++j)
          {
            double sum_laplace = 0;
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              {
                sum_laplace += (fe_1d->shape_grad(i, quadrature.point(q))[0] *
                                fe_1d->shape_grad(j, quadrature.point(q))[0]) *
                               quadrature.weight(q) * is_first;
              }

            sum_laplace +=
              boundary_factor_left *
              (1. * fe_1d->shape_value(i, Point<1>()) *
                 fe_1d->shape_value(j, Point<1>()) * penalty_factor +
               0.5 * fe_1d->shape_grad(i, Point<1>())[0] *
                 fe_1d->shape_value(j, Point<1>()) +
               0.5 * fe_1d->shape_grad(j, Point<1>())[0] *
                 fe_1d->shape_value(i, Point<1>()));

            sum_laplace +=
              boundary_factor_right *
              (1. * fe_1d->shape_value(i, Point<1>(1.0)) *
                 fe_1d->shape_value(j, Point<1>(1.0)) * penalty_factor -
               0.5 * fe_1d->shape_grad(i, Point<1>(1.0))[0] *
                 fe_1d->shape_value(j, Point<1>(1.0)) -
               0.5 * fe_1d->shape_grad(j, Point<1>(1.0))[0] *
                 fe_1d->shape_value(i, Point<1>(1.0)));

            // scaling to real cells
            cell_laplace(i, j) = sum_laplace * scaling_factor;
          }

      return cell_laplace;
    };

    for (unsigned int i = 0; i < N; ++i)
      for (unsigned int j = 0; j < N; ++j)
        {
          double sum_mass = 0, sum_mixed = 0, sum_penalty = 0;
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            {
              sum_mass += (fe_1d->shape_value(i, quadrature.point(q)) *
                           fe_1d->shape_value(j, quadrature.point(q))) *
                          quadrature.weight(q);
            }
          patch_mass_0(i, j)         = sum_mass;
          patch_mass_1(i, j)         = sum_mass;
          patch_mass_1(i + N, j + N) = sum_mass;

          sum_mixed += (-0.5 * fe_1d->shape_grad(i, Point<1>())[0] *
                        fe_1d->shape_value(j, Point<1>(1.0)));

          sum_penalty +=
            (-1. * fe_1d->shape_value(i, Point<1>()) *
             fe_1d->shape_value(j, Point<1>(1.0)) * penalty_factor);

          laplace_interface_mixed(N - 1 - i, N - 1 - j) =
            scaling_factor * sum_mixed;
          laplace_interface_penalty(N - 1 - i, N - 1 - j) =
            scaling_factor * sum_penalty;
        }

    auto laplace_left     = get_cell_laplace(0, 0);
    auto laplace_middle_0 = get_cell_laplace(1, 0);
    auto laplace_middle_1 = get_cell_laplace(1, 1);
    auto laplace_right    = get_cell_laplace(2, 0);

    // mass, laplace
    auto get_patch_laplace = [&](auto left, auto right) {
      Table<2, Number> patch_laplace;
      patch_laplace.reinit(2 * N, 2 * N);

      for (unsigned int i = 0; i < N; ++i)
        for (unsigned int j = 0; j < N; ++j)
          {
            patch_laplace(i, j)         = left(i, j);
            patch_laplace(i + N, j + N) = right(i, j);

            patch_laplace(i, j + N) = laplace_interface_mixed(i, j);
            patch_laplace(i, j + N) +=
              laplace_interface_mixed(N - 1 - j, N - 1 - i);
            patch_laplace(i, j + N) +=
              laplace_interface_penalty(N - 1 - j, N - 1 - i);
            patch_laplace(j + N, i) = patch_laplace(i, j + N);
          }

      return patch_laplace;
    };

    auto patch_laplace_0 = get_patch_laplace(laplace_left, laplace_middle_1);
    auto patch_laplace_1 =
      get_patch_laplace(laplace_middle_0, laplace_middle_1);
    auto patch_laplace_2 = get_patch_laplace(laplace_middle_0, laplace_right);

    if (level == 1)
      patch_laplace_2 = get_patch_laplace(laplace_left, laplace_right);

    constexpr unsigned int n_dofs_2d = Util::pow(2 * fe_degree + 2, 2);

    auto *mass    = new Number[n_dofs_2d * 3];
    auto *laplace = new Number[n_dofs_2d * 3];

    std::transform(patch_mass_0.begin(),
                   patch_mass_0.end(),
                   &mass[n_dofs_2d * 0],
                   [](const Number m) -> Number { return m; });

    std::transform(patch_mass_0.begin(),
                   patch_mass_0.end(),
                   &mass[n_dofs_2d * 1],
                   [](const Number m) -> Number { return m; });

    std::transform(patch_mass_1.begin(),
                   patch_mass_1.end(),
                   &mass[n_dofs_2d * 2],
                   [](const Number m) -> Number { return m; });

    std::transform(patch_laplace_0.begin(),
                   patch_laplace_0.end(),
                   &laplace[n_dofs_2d * 0],
                   [](const Number m) -> Number { return m; });

    std::transform(patch_laplace_1.begin(),
                   patch_laplace_1.end(),
                   &laplace[n_dofs_2d * 1],
                   [](const Number m) -> Number { return m; });

    std::transform(patch_laplace_2.begin(),
                   patch_laplace_2.end(),
                   &laplace[n_dofs_2d * 2],
                   [](const Number m) -> Number { return m; });


    cudaError_t error_code = cudaMemcpy(global_mass_1d,
                                        mass,
                                        3 * n_dofs_2d * sizeof(Number),
                                        cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    error_code = cudaMemcpy(global_derivative_1d,
                            laplace,
                            3 * n_dofs_2d * sizeof(Number),
                            cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    // for (unsigned int i = 0; i < 3 * n_dofs_2d; ++i)
    //   std::cout << mass[i] << " " << laplace[i] << std::endl;

    delete[] mass;
    delete[] laplace;
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::DGQ>::
    setup_color_arrays(const unsigned int n_colors)
  {
    this->n_patches.resize(n_colors);
    this->grid_dim.resize(n_colors);
    this->block_dim.resize(n_colors);
    this->block_dim_inv.resize(n_colors);
    this->first_dof.resize(n_colors);
    this->patch_id.resize(n_colors);
    this->patch_type.resize(n_colors);
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::DGQ>::
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

    constexpr unsigned int n_dofs_1d     = 2 * fe_degree + 2;
    constexpr unsigned int n_dofs_1d_inv = 2 * fe_degree - 1;

    switch (kernel)
      {
        case SmootherVariant::FUSED_BASE:
          block_dim[color] = dim3(patch_per_block * n_dofs_1d, n_dofs_1d);
          break;
        case SmootherVariant::FUSED_L:
          block_dim[color] = dim3(patch_per_block * n_dofs_1d, n_dofs_1d);
          break;
        case SmootherVariant::FUSED_3D:
          Assert(fe_degree < 5, ExcNotImplemented());
          AssertDimension(dim, 3);
          block_dim[color] =
            dim3(patch_per_block * n_dofs_1d, n_dofs_1d, n_dofs_1d);
          break;
        case SmootherVariant::FUSED_CF:
          Assert(dim == 2 || fe_degree < 8, ExcNotImplemented());
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
  }

  template <int dim, int fe_degree, typename Number, SmootherVariant kernel>
  template <typename Number1>
  void
  LevelVertexPatch<dim, fe_degree, Number, kernel, DoFLayout::DGQ>::
    alloc_arrays(Number1 **array_device, const unsigned int n)
  {
    cudaError_t error_code = cudaMalloc(array_device, n * sizeof(Number1));
    AssertCuda(error_code);
  }

} // namespace PSMF

/**
 * \page patch_base.template
 * \include patch_base.template.cuh
 */