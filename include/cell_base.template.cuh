/**
 * @file cell_base.template.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief This class collects all the data that is stored for the matrix free implementation.
 * @version 1.0
 * @date 2023-02-02
 *
 * @copyright Copyright (c) 2023
 *
 */


namespace PSMF
{
  template <int dim, int fe_degree, typename Number>
  __device__ bool
  LevelCellPatch<dim, fe_degree, Number>::Data::is_ghost(
    const unsigned int global_index) const
  {
    return !(local_range_start <= global_index &&
             global_index < local_range_end);
  }

  template <int dim, int fe_degree, typename Number>
  __device__ types::global_dof_index
             LevelCellPatch<dim, fe_degree, Number>::Data::global_to_local(
    const types::global_dof_index global_index) const
  {
    if (local_range_start <= global_index && global_index < local_range_end)
      return global_index - local_range_start;
    else
      {
        printf("*************** ERROR index: %lu ***************\n",
               global_index);
        printf("******** All indices should be local **********\n");

        const unsigned int index_within_ghosts =
          binary_search(global_index, 0, n_ghost_indices - 1);

        return local_range_end - local_range_start + index_within_ghosts;
      }
  }

  template <int dim, int fe_degree, typename Number>
  __device__ unsigned int
  LevelCellPatch<dim, fe_degree, Number>::Data::binary_search(
    const unsigned int local_index,
    const unsigned int l,
    const unsigned int r) const
  {
    if (r >= l)
      {
        unsigned int mid = l + (r - l) / 2;

        if (ghost_indices[mid] == local_index)
          return mid;

        if (ghost_indices[mid] > local_index)
          return binary_search(local_index, l, mid - 1);

        return binary_search(local_index, mid + 1, r);
      }

    printf("*************** ERROR index: %d ***************\n", local_index);
    return 0;
  }

  template <int dim, int fe_degree, typename Number>
  LevelCellPatch<dim, fe_degree, Number>::LevelCellPatch()
  {}

  template <int dim, int fe_degree, typename Number>
  LevelCellPatch<dim, fe_degree, Number>::~LevelCellPatch()
  {
    free();
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelCellPatch<dim, fe_degree, Number>::free()
  {
    for (auto &first_dof_color_ptr : first_dof_smooth)
      Utilities::CUDA::free(first_dof_color_ptr);
    first_dof_smooth.clear();
    first_dof_smooth.shrink_to_fit();

    cell_type.clear();
    cell_type.shrink_to_fit();

    Utilities::CUDA::free(smooth_mass_1d);
    Utilities::CUDA::free(smooth_stiff_1d);
    Utilities::CUDA::free(eigenvalues);
    Utilities::CUDA::free(eigenvectors);

    cell_type_host.clear();
    first_dof_host.clear();

    cell_type_host.shrink_to_fit();
    first_dof_host.shrink_to_fit();

    AssertCuda(cudaStreamDestroy(stream));
    AssertCuda(cudaStreamDestroy(stream_g));
  }

  template <int dim, int fe_degree, typename Number>
  std::size_t
  LevelCellPatch<dim, fe_degree, Number>::memory_consumption() const
  {
    const unsigned int n_dofs_1d = fe_degree + 1;

    std::size_t result = 0;

    // For each color, add first_dof, cell_id, {mass,derivative}_matrix,
    // and eigen{values,vectors}.
    for (unsigned int i = 0; i < n_colors; ++i)
      {
        result += 2 * n_cells_laplace[i] * sizeof(unsigned int) +
                  2 * n_dofs_1d * n_dofs_1d * (1 << level) * sizeof(Number) +
                  2 * n_dofs_1d * dim * sizeof(Number);
      }
    return result;
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelCellPatch<dim, fe_degree, Number>::get_cell_data(
    const CellIterator           &cell,
    const types::global_dof_index cell_id,
    const bool                    is_ghost)
  {
    std::vector<types::global_dof_index> local_dof_indices(
      Util::pow(fe_degree + 1, dim));

    // first_dof
    cell->get_mg_dof_indices(local_dof_indices);

    first_dof_host[cell_id] = local_dof_indices[0];


    // cell_type. TODO: Fix: only works on [0,1]^d
    const double h            = 1. / Util::pow(2, level);
    auto         first_center = cell->center();

    for (unsigned int d = 0; d < dim; ++d)
      {
        auto scale = d == 0 ? n_replicate : 1;
        auto pos   = std::floor(first_center[d] / h + 1 / 3);
        cell_type_host[cell_id * dim + d] =
          (pos > 0) + (pos == (scale * Util::pow(2, level) - 1));
      }
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelCellPatch<dim, fe_degree, Number>::reinit(
    const DoFHandler<dim> &mg_dof,
    const unsigned int     mg_level,
    const AdditionalData  &additional_data)
  {
    if (typeid(Number) == typeid(double))
      cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    this->relaxation         = additional_data.relaxation;
    this->granularity_scheme = additional_data.granularity_scheme;

    dof_handler = &mg_dof;
    level       = mg_level;

    n_replicate = dof_handler->get_triangulation().n_cells(0);

    auto locally_owned_dofs = dof_handler->locally_owned_mg_dofs(level);
    auto locally_relevant_dofs =
      DoFTools::extract_locally_relevant_level_dofs(*dof_handler, level);

    partitioner =
      std::make_shared<Utilities::MPI::Partitioner>(locally_owned_dofs,
                                                    locally_relevant_dofs,
                                                    MPI_COMM_WORLD);

    switch (granularity_scheme)
      {
        case GranularityScheme::none:
          cell_per_block = 1;
          break;
        case GranularityScheme::user_define:
          cell_per_block = additional_data.cell_per_block;
          break;
        case GranularityScheme::multiple:
          cell_per_block = cell_granularity_shmem<dim, fe_degree>();
          break;
        default:
          AssertThrow(false, ExcMessage("Invalid granularity scheme."));
          break;
      }

    // red-black coloring
    if (mg_level == 0)
      {
        n_colors = 1;
        graph_ptr_colored.clear();
        graph_ptr_colored.resize(n_colors);

        // TODO: root cells > 1
        auto cell = mg_dof.begin_mg(mg_level);
        if (cell->is_locally_owned_on_level())
          graph_ptr_colored[0].push_back(cell);
      }
    else
      {
        n_colors = 2;
        graph_ptr_colored.clear();
        graph_ptr_colored.resize(n_colors);

        for (auto cell = mg_dof.begin_mg(mg_level);
             cell != mg_dof.end_mg(mg_level);
             ++cell)
          if (cell->is_locally_owned_on_level())
            {
              unsigned int index =
                cell->parent()->child_iterator_to_index(cell);
              if (index == 0 || index == 3 || index == 5 || index == 6)
                graph_ptr_colored[0].push_back(cell);
              else
                graph_ptr_colored[1].push_back(cell);
            }
      }

    // if (level == 2)
    //   {
    //     for (auto cell : graph_ptr_colored[0])
    //       std::cout << cell << std::endl;
    //     std::cout << std::endl;
    //     for (auto cell : graph_ptr_colored[1])
    //       std::cout << cell << std::endl;
    //   }

    setup_color_arrays(n_colors);

    for (unsigned int i = 0; i < n_colors; ++i)
      {
        auto n_cells      = graph_ptr_colored[i].size();
        n_cells_smooth[i] = n_cells;

        cell_type_host.clear();
        first_dof_host.clear();
        cell_type_host.resize(n_cells * dim);
        first_dof_host.resize(n_cells);

        auto cell      = graph_ptr_colored[i].begin(),
             end_patch = graph_ptr_colored[i].end();
        for (types::global_dof_index c_id = 0; cell != end_patch;
             ++cell, ++c_id)
          get_cell_data(*cell, c_id, false);

        alloc_arrays(&first_dof_smooth[i], n_cells);
        alloc_arrays(&cell_type[i], n_cells * dim);

        cudaError_t error_code =
          cudaMemcpy(first_dof_smooth[i],
                     first_dof_host.data(),
                     n_cells * sizeof(types::global_dof_index),
                     cudaMemcpyHostToDevice);
        AssertCuda(error_code);

        error_code = cudaMemcpy(cell_type[i],
                                cell_type_host.data(),
                                n_cells * dim * sizeof(unsigned int),
                                cudaMemcpyHostToDevice);
        AssertCuda(error_code);
      }

    setup_configuration(n_colors);

    constexpr unsigned int n_dofs_1d = fe_degree + 1;
    constexpr unsigned int n_dofs_2d = n_dofs_1d * n_dofs_1d;

    alloc_arrays(&eigenvalues, n_dofs_1d * dim * std::pow(3, dim));
    alloc_arrays(&eigenvectors, n_dofs_2d * dim * std::pow(3, dim));
    alloc_arrays(&smooth_mass_1d, n_dofs_2d * 3);
    alloc_arrays(&smooth_stiff_1d, n_dofs_2d * 3);

    reinit_tensor_product_smoother();

    AssertCuda(cudaStreamCreate(&stream));
    AssertCuda(cudaStreamCreate(&stream_g));

    // ghost
    auto ghost_indices = partitioner->ghost_indices();
    auto local_range   = partitioner->local_range();
    n_ghost_indices    = ghost_indices.n_elements();

    local_range_start = local_range.first;
    local_range_end   = local_range.second;

    auto *ghost_indices_host = new types::global_dof_index[n_ghost_indices];
    for (types::global_dof_index i = 0; i < n_ghost_indices; ++i)
      ghost_indices_host[i] = ghost_indices.nth_index_in_set(i);

    alloc_arrays(&ghost_indices_dev, n_ghost_indices);
    cudaError_t error_code =
      cudaMemcpy(ghost_indices_dev,
                 ghost_indices_host,
                 n_ghost_indices * sizeof(types::global_dof_index),
                 cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    solution_ghosted = std::make_shared<
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>>();
    solution_ghosted->reinit(partitioner);

    cell_type_host.clear();
    first_dof_host.clear();

    cell_type_host.shrink_to_fit();
    first_dof_host.shrink_to_fit();

    delete[] ghost_indices_host;
  }

  template <int dim, int fe_degree, typename Number>
  LevelCellPatch<dim, fe_degree, Number>::Data
  LevelCellPatch<dim, fe_degree, Number>::get_smooth_data(
    unsigned int color) const
  {
    Data data_copy;

    data_copy.n_cells         = n_cells_smooth[color];
    data_copy.cell_per_block  = cell_per_block;
    data_copy.relaxation      = relaxation;
    data_copy.first_dof       = first_dof_smooth[color];
    data_copy.cell_type       = cell_type[color];
    data_copy.eigenvalues     = eigenvalues;
    data_copy.eigenvectors    = eigenvectors;
    data_copy.smooth_mass_1d  = smooth_mass_1d;
    data_copy.smooth_stiff_1d = smooth_stiff_1d;

    data_copy.n_ghost_indices   = n_ghost_indices;
    data_copy.local_range_start = local_range_start;
    data_copy.local_range_end   = local_range_end;
    data_copy.ghost_indices     = ghost_indices_dev;

    return data_copy;
  }

  template <int dim, int fe_degree, typename Number>
  template <typename Operator, typename VectorType>
  void
  LevelCellPatch<dim, fe_degree, Number>::cell_loop(const Operator   &op,
                                                    const VectorType &src,
                                                    VectorType       &dst) const
  {
    Util::adjust_ghost_range_if_necessary(src, partitioner);
    Util::adjust_ghost_range_if_necessary(dst, partitioner);

    src.update_ghost_values();

    for (unsigned int i = 0; i < n_colors; ++i)
      {
        op.template setup_kernel<false>(cell_per_block);

        // if (n_cells_smooth[i] > 0)
        {
          op.template loop_kernel<VectorType, Data, false>(src,
                                                           dst,
                                                           dst,
                                                           get_smooth_data(i),
                                                           grid_dim_smooth[i],
                                                           block_dim_smooth[i],
                                                           stream);

          AssertCudaKernel();
        }

        // dst.compress(VectorOperation::add);
      }
    src.zero_out_ghost_values();
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelCellPatch<dim, fe_degree, Number>::reinit_tensor_product_smoother() const
  {
    std::string name = dof_handler->get_fe().get_name();
    name.replace(name.find('<') + 1, 1, "1");
    std::unique_ptr<FiniteElement<1>> fe_1d = FETools::get_fe_by_name<1>(name);

    constexpr unsigned int N              = fe_degree + 1;
    constexpr Number       penalty_factor = 1.0 * fe_degree * (fe_degree + 1);
    const Number scaling_factor = dim == 2 ? 1 : 1. / Util::pow(2, level);

    QGauss<1> quadrature(N);

    std::array<Table<2, Number>, 1> cell_mass;
    std::array<Table<2, Number>, 3> cell_stiffness;

    cell_mass[0].reinit(N, N);
    for (unsigned int d = 0; d < 3; ++d)
      cell_stiffness[d].reinit(N, N);


    auto get_cell_laplace = [&](unsigned int type) {
      Number boundary_factor_left  = 1.;
      Number boundary_factor_right = 1.;

      if (type == 0)
        boundary_factor_left = 2.;
      else if (type == 1)
        {
        }
      else if (type == 2)
        boundary_factor_right = 2.;
      else
        Assert(false, ExcNotImplemented());

      if (level == 0)
        {
          boundary_factor_left  = 2.;
          boundary_factor_right = 2.;
        }

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
            cell_stiffness[type](i, j) = sum_laplace * scaling_factor;
          }
    };

    for (unsigned int d = 0; d < 1; ++d)
      for (unsigned int i = 0; i < N; ++i)
        for (unsigned int j = 0; j < N; ++j)
          {
            double sum_mass = 0;
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              {
                sum_mass += (fe_1d->shape_value(i, quadrature.point(q)) *
                             fe_1d->shape_value(j, quadrature.point(q))) *
                            quadrature.weight(q);
              }

            cell_mass[d](i, j) = sum_mass;
          }

    for (unsigned int d = 0; d < 3; ++d)
      get_cell_laplace(d);

    auto *mass    = new Number[N * N];
    auto *laplace = new Number[N * N * 3];
    for (int d = 0; d < 3; ++d)
      {
        if (d == 0)
          std::transform(cell_mass[d].begin(),
                         cell_mass[d].end(),
                         &mass[N * N * d],
                         [](const Number m) -> Number { return m; });

        std::transform(cell_stiffness[d].begin(),
                       cell_stiffness[d].end(),
                       &laplace[N * N * d],
                       [](const Number m) -> Number { return m; });
      }
    cudaError_t error_code = cudaMemcpy(smooth_mass_1d,
                                        mass,
                                        N * N * sizeof(Number),
                                        cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    error_code = cudaMemcpy(smooth_stiff_1d,
                            laplace,
                            N * N * 3 * sizeof(Number),
                            cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    delete[] mass;
    delete[] laplace;

    // if (level == 2)
    //   {
    //     auto print_matrices = [](auto matrix) {
    //       for (auto m = 0U; m < matrix.size(0); ++m)
    //         {
    //           for (auto n = 0U; n < matrix.size(1); ++n)
    //             std::cout << matrix(m, n) << " ";
    //           std::cout << std::endl;
    //         }
    //       std::cout << std::endl;
    //     };
    //
    //     print_matrices(cell_mass[0]);
    //     print_matrices(cell_stiffness[1]);
    //   }

    auto copy_vals = [](auto tensor, auto dst, auto shift) {
      constexpr unsigned int n_dofs_1d = Util::pow(fe_degree + 1, 1);

      auto mat = new Number[n_dofs_1d * dim];
      for (unsigned int i = 0; i < dim; ++i)
        std::transform(tensor[i].begin(),
                       tensor[i].end(),
                       &mat[n_dofs_1d * i],
                       [](auto m) -> Number { return m; });

      cudaError_t error_code = cudaMemcpy(dst + shift * n_dofs_1d * dim,
                                          mat,
                                          dim * n_dofs_1d * sizeof(Number),
                                          cudaMemcpyHostToDevice);
      AssertCuda(error_code);

      delete[] mat;
    };

    auto copy_vecs = [](auto tensor, auto dst, auto shift) {
      constexpr unsigned int n_dofs_2d = Util::pow(fe_degree + 1, 2);

      auto mat = new Number[n_dofs_2d * dim];
      for (unsigned int i = 0; i < dim; ++i)
        std::transform(tensor[i].begin(),
                       tensor[i].end(),
                       &mat[n_dofs_2d * i],
                       [](auto m) -> Number { return m; });

      cudaError_t error_code = cudaMemcpy(dst + shift * n_dofs_2d * dim,
                                          mat,
                                          dim * n_dofs_2d * sizeof(Number),
                                          cudaMemcpyHostToDevice);
      AssertCuda(error_code);

      delete[] mat;
    };

    auto fast_diag = [&](auto indices) {
      std::array<Table<2, Number>, dim> cell_mass_inv;
      std::array<Table<2, Number>, dim> cell_laplace_inv;

      for (unsigned int d = 0; d < dim; ++d)
        {
          cell_mass_inv[d]    = cell_mass[0];
          cell_laplace_inv[d] = cell_stiffness[indices[d]];
        }

      TensorProductData<dim, fe_degree, Number> tensor_product;
      tensor_product.reinit(cell_mass_inv, cell_laplace_inv);

      std::array<AlignedVector<Number>, dim> eigenvalue_tensor;
      std::array<Table<2, Number>, dim>      eigenvector_tensor;
      tensor_product.get_eigenvalues(eigenvalue_tensor);
      tensor_product.get_eigenvectors(eigenvector_tensor);

      auto shift = indices[0] + indices[1] * 3 + indices[2] * 9;

      copy_vals(eigenvalue_tensor, eigenvalues, shift);
      copy_vecs(eigenvector_tensor, eigenvectors, shift);
    };

    constexpr unsigned dim_z = dim == 2 ? 1 : 3;

#pragma omp parallel for collapse(3) num_threads(dim_z * 3 * 3) schedule(static)
    for (unsigned int z = 0; z < dim_z; ++z)
      for (unsigned int j = 0; j < 3; ++j)
        for (unsigned int k = 0; k < 3; ++k)
          {
            std::vector<unsigned int> indices{k, j, z};
            fast_diag(indices);
          }
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelCellPatch<dim, fe_degree, Number>::setup_color_arrays(
    const unsigned int n_colors)
  {
    this->n_cells_laplace.resize(n_colors);
    this->grid_dim_lapalce.resize(n_colors);
    this->block_dim_laplace.resize(n_colors);
    this->first_dof_laplace.resize(n_colors);
    this->cell_type.resize(n_colors);

    this->n_cells_smooth.resize(n_colors);
    this->grid_dim_smooth.resize(n_colors);
    this->block_dim_smooth.resize(n_colors);
    this->first_dof_smooth.resize(n_colors);
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelCellPatch<dim, fe_degree, Number>::setup_configuration(
    const unsigned int n_colors)
  {
    constexpr unsigned int n_dofs_1d = fe_degree + 1;

    for (unsigned int i = 0; i < n_colors; ++i)
      {
        auto   n_cells        = n_cells_laplace[i];
        double apply_n_blocks = std::ceil(static_cast<double>(n_cells) /
                                          static_cast<double>(cell_per_block));

        grid_dim_lapalce[i]  = dim3(apply_n_blocks);
        block_dim_laplace[i] = dim3(cell_per_block * n_dofs_1d, n_dofs_1d);

        // n_cells        = n_cells_laplace_ghost[i];
        // apply_n_blocks = std::ceil(static_cast<double>(n_cells) /
        //                            static_cast<double>(cell_per_block));
        //
        // grid_dim_lapalce_ghost[i] = dim3(apply_n_blocks);
      }

    for (unsigned int i = 0; i < n_colors; ++i)
      {
        auto   n_cells        = n_cells_smooth[i];
        double apply_n_blocks = std::ceil(static_cast<double>(n_cells) /
                                          static_cast<double>(cell_per_block));

        grid_dim_smooth[i]  = dim3(apply_n_blocks);
        block_dim_smooth[i] = dim3(cell_per_block * n_dofs_1d, n_dofs_1d);

        // n_cells        = n_cells_smooth_ghost[i];
        // apply_n_blocks = std::ceil(static_cast<double>(n_cells) /
        //                            static_cast<double>(cell_per_block));
        //
        // grid_dim_smooth_ghost[i] = dim3(apply_n_blocks);
      }
  }

  template <int dim, int fe_degree, typename Number>
  template <typename Number1>
  void
  LevelCellPatch<dim, fe_degree, Number>::alloc_arrays(
    Number1                     **array_device,
    const types::global_dof_index n)
  {
    cudaError_t error_code = cudaMalloc(array_device, n * sizeof(Number1));
    AssertCuda(error_code);
  }

} // namespace PSMF

/**
 * \page cell_base.template
 * \include cell_base.template.cuh
 */
