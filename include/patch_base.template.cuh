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
#include "renumbering.h"

namespace PSMF
{
  template <int dim, int fe_degree, typename Number>
  __device__ bool
  LevelVertexPatch<dim, fe_degree, Number>::Data::is_ghost(
    const unsigned int global_index) const
  {
    return !(local_range_start <= global_index &&
             global_index < local_range_end);
  }

  template <int dim, int fe_degree, typename Number>
  __device__ unsigned int
  LevelVertexPatch<dim, fe_degree, Number>::Data::global_to_local(
    const unsigned int global_index) const
  {
    if (local_range_start <= global_index && global_index < local_range_end)
      return global_index - local_range_start;
    else
      {
        printf("*************** ERROR index: %d ***************\n",
               global_index);
        printf("******** All indices should be local **********\n");

        const unsigned int index_within_ghosts =
          binary_search(global_index, 0, n_ghost_indices - 1);

        return local_range_end - local_range_start + index_within_ghosts;
      }
  }

  template <int dim, int fe_degree, typename Number>
  __device__ unsigned int
  LevelVertexPatch<dim, fe_degree, Number>::Data::binary_search(
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
  LevelVertexPatch<dim, fe_degree, Number>::GhostPatch::GhostPatch(
    const unsigned int proc,
    const CellId      &cell_id)
  {
    submit_id(proc, cell_id);
  }


  template <int dim, int fe_degree, typename Number>
  inline void
  LevelVertexPatch<dim, fe_degree, Number>::GhostPatch::submit_id(
    const unsigned int proc,
    const CellId      &cell_id)
  {
    const auto member = proc_to_cell_ids.find(proc);
    if (member != proc_to_cell_ids.cend())
      {
        member->second.emplace_back(cell_id);
        Assert(!(member->second.empty()), ExcMessage("at least one element"));
      }
    else
      {
        const auto status =
          proc_to_cell_ids.emplace(proc, std::vector<CellId>{cell_id});
        (void)status;
        Assert(status.second, ExcMessage("failed to insert key-value-pair"));
      }
  }

  template <int dim, int fe_degree, typename Number>
  LevelVertexPatch<dim, fe_degree, Number>::LevelVertexPatch()
  {}

  template <int dim, int fe_degree, typename Number>
  LevelVertexPatch<dim, fe_degree, Number>::~LevelVertexPatch()
  {
    free();
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelVertexPatch<dim, fe_degree, Number>::free()
  {
    for (auto &first_dof_color_ptr : first_dof_laplace)
      Utilities::CUDA::free(first_dof_color_ptr);
    first_dof_laplace.clear();
    first_dof_laplace.shrink_to_fit();

    for (auto &first_dof_color_ptr : first_dof_smooth)
      Utilities::CUDA::free(first_dof_color_ptr);
    first_dof_smooth.clear();
    first_dof_smooth.shrink_to_fit();

    for (auto &patch_id_color_ptr : patch_id)
      Utilities::CUDA::free(patch_id_color_ptr);
    patch_id.clear();
    patch_id.shrink_to_fit();

    for (auto &patch_type_color_ptr : patch_type)
      Utilities::CUDA::free(patch_type_color_ptr);
    patch_type.clear();
    patch_type.shrink_to_fit();

    Utilities::CUDA::free(laplace_mass_1d);
    Utilities::CUDA::free(laplace_stiff_1d);
    Utilities::CUDA::free(smooth_mass_1d);
    Utilities::CUDA::free(smooth_stiff_1d);
    Utilities::CUDA::free(eigenvalues);
    Utilities::CUDA::free(eigenvectors);


    ordering_to_type.clear();
    patch_id_host.clear();
    patch_type_host.clear();
    first_dof_host.clear();
    h_to_l_host.clear();
    l_to_h_host.clear();

    patch_id_host.shrink_to_fit();
    patch_type_host.shrink_to_fit();
    first_dof_host.shrink_to_fit();
    h_to_l_host.shrink_to_fit();
    l_to_h_host.shrink_to_fit();

    AssertCuda(cudaStreamDestroy(stream));
    AssertCuda(cudaStreamDestroy(stream_g));
  }

  template <int dim, int fe_degree, typename Number>
  std::size_t
  LevelVertexPatch<dim, fe_degree, Number>::memory_consumption() const
  {
    const unsigned int n_dofs_1d = 2 * fe_degree + 2;

    std::size_t result = 0;

    // For each color, add first_dof, patch_id, {mass,derivative}_matrix,
    // and eigen{values,vectors}.
    for (unsigned int i = 0; i < n_colors; ++i)
      {
        result += 2 * n_patches_laplace[i] * sizeof(unsigned int) +
                  2 * n_dofs_1d * n_dofs_1d * (1 << level) * sizeof(Number) +
                  2 * n_dofs_1d * dim * sizeof(Number);
      }
    return result;
  }

  template <int dim, int fe_degree, typename Number>
  std::vector<std::vector<
    typename LevelVertexPatch<dim, fe_degree, Number>::CellIterator>>
  LevelVertexPatch<dim, fe_degree, Number>::gather_vertex_patches(
    const DoFHandler<dim> &dof_handler,
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

    const auto &tria = dof_handler.get_triangulation();
    const auto  locally_owned_range_mg =
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
     * Ghost patches are stored as the mapping @p global_to_ghost_id
     * between the global vertex index and GhostPatch. The number of
     * cells, book-kept in @p global_to_local_map, is updated taking the
     * ghost cells into account.
     */
    // TODO: is_ghost_on_level() missing
    const auto not_locally_owned_range_mg =
      filter_iterators(dof_handler.mg_cell_iterators_on_level(level),
                       [](const auto &cell) {
                         return !(cell->is_locally_owned_on_level());
                       });
    std::map<unsigned int, GhostPatch> global_to_ghost_id;
    for (const auto &cell : not_locally_owned_range_mg)
      {
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            const unsigned int global_index = cell->vertex_index(v);
            const auto         element = global_to_local_map.find(global_index);
            if (element != global_to_local_map.cend())
              {
                ++(element->second.second);
                const unsigned int subdomain_id_ghost =
                  cell->level_subdomain_id();
                const auto ghost = global_to_ghost_id.find(global_index);
                if (ghost != global_to_ghost_id.cend())
                  ghost->second.submit_id(subdomain_id_ghost, cell->id());
                else
                  {
                    const auto status = global_to_ghost_id.emplace(
                      global_index, GhostPatch(subdomain_id_ghost, cell->id()));
                    (void)status;
                    Assert(status.second,
                           ExcMessage("failed to insert key-value-pair"));
                  }
              }
          }
      }

    { // ASSIGN GHOSTS
      const unsigned int my_subdomain_id = tria.locally_owned_subdomain();
      /**
       * logic: if the mpi-proc owns more than half of the cells within
       *        a ghost patch he takes ownership
       */
      {
        //: (1) add subdomain_ids of locally owned cells to GhostPatches
        for (const auto &cell : locally_owned_range_mg)
          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
               ++v)
            {
              const unsigned global_index = cell->vertex_index(v);
              const auto     ghost = global_to_ghost_id.find(global_index);
              //: checks if the global vertex has ghost cells attached
              if (ghost != global_to_ghost_id.end())
                ghost->second.submit_id(my_subdomain_id, cell->id());
            }

        std::set<unsigned> to_be_owned;
        std::set<unsigned> to_be_erased;
        for (const auto &key_value : global_to_ghost_id)
          {
            const unsigned int global_index = key_value.first;
            const auto &proc_to_cell_ids    = key_value.second.proc_to_cell_ids;

            const auto &get_proc_with_most_cellids = [](const auto &lhs,
                                                        const auto &rhs) {
              const std::vector<CellId> &cell_ids_lhs = lhs.second;
              const std::vector<CellId> &cell_ids_rhs = rhs.second;
              Assert(!cell_ids_lhs.empty(), ExcMessage("should not be empty"));
              Assert(!cell_ids_rhs.empty(), ExcMessage("should not be empty"));
              return (cell_ids_lhs.size() < cell_ids_rhs.size());
            };

            const auto most = std::max_element(proc_to_cell_ids.cbegin(),
                                               proc_to_cell_ids.cend(),
                                               get_proc_with_most_cellids);
            const unsigned int subdomain_id_most          = most->first;
            const unsigned int n_locally_owned_cells_most = most->second.size();
            const auto         member = global_to_local_map.find(global_index);
            Assert(member != global_to_local_map.cend(),
                   ExcMessage("must be listed as patch"));
            const unsigned int n_cells = member->second.second;
            if (my_subdomain_id == subdomain_id_most)
              {
                AssertDimension(member->second.first,
                                n_locally_owned_cells_most);
                if (2 * n_locally_owned_cells_most > n_cells)
                  to_be_owned.insert(global_index);
              }
            else
              {
                if (2 * n_locally_owned_cells_most > n_cells)
                  to_be_erased.insert(global_index);
              }
          }

        for (const unsigned global_index : to_be_owned)
          {
            auto &my_patch = global_to_local_map[global_index];
            my_patch.first = my_patch.second;
            global_to_ghost_id.erase(global_index);
          }
        for (const unsigned global_index : to_be_erased)
          {
            global_to_local_map.erase(global_index);
            global_to_ghost_id.erase(global_index);
          }
      }

      /**
       * logic: the owner of the cell with the lowest CellId takes ownership
       */
      {
        //: (2) determine mpi-proc with the minimal CellId for all GhostPatches
        std::set<unsigned> to_be_owned;
        for (const auto &key_value : global_to_ghost_id)
          {
            const unsigned int global_index = key_value.first;
            const auto &proc_to_cell_ids    = key_value.second.proc_to_cell_ids;

            const auto &get_proc_with_min_cellid = [](const auto &lhs,
                                                      const auto &rhs) {
              std::vector<CellId> cell_ids_lhs = lhs.second;
              Assert(!cell_ids_lhs.empty(), ExcMessage("should not be empty"));
              std::sort(cell_ids_lhs.begin(), cell_ids_lhs.end());
              const auto          min_cell_id_lhs = cell_ids_lhs.front();
              std::vector<CellId> cell_ids_rhs    = rhs.second;
              Assert(!cell_ids_rhs.empty(), ExcMessage("should not be empty"));
              std::sort(cell_ids_rhs.begin(), cell_ids_rhs.end());
              const auto min_cell_id_rhs = cell_ids_rhs.front();
              return min_cell_id_lhs < min_cell_id_rhs;
            };

            const auto min = std::min_element(proc_to_cell_ids.cbegin(),
                                              proc_to_cell_ids.cend(),
                                              get_proc_with_min_cellid);

            const unsigned int subdomain_id_min = min->first;
            if (my_subdomain_id == subdomain_id_min)
              to_be_owned.insert(global_index);
          }

        //: (3) set owned GhostPatches in global_to_local_map and delete all
        //: remaining
        for (const unsigned global_index : to_be_owned)
          {
            auto &my_patch = global_to_local_map[global_index];
            my_patch.first = my_patch.second;
            global_to_ghost_id.erase(global_index);
          }
        for (const auto &key_value : global_to_ghost_id)
          {
            const unsigned int global_index = key_value.first;
            global_to_local_map.erase(global_index);
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
              else                                   // irregular patch
                AssertThrow(false, ExcMessage("TODO irregular vertex patches"));
            }
        }
    return cell_collections;
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelVertexPatch<dim, fe_degree, Number>::get_patch_data(
    const PatchIterator &patch,
    const unsigned int   patch_id,
    const bool           is_ghost)
  {
    std::vector<unsigned int> local_dof_indices(Util::pow(fe_degree + 1, dim));
    std::vector<unsigned int> numbering(regular_vpatch_size);
    std::iota(numbering.begin(), numbering.end(), 0);

    // first_dof
    for (unsigned int cell = 0; cell < regular_vpatch_size; ++cell)
      {
        auto cell_ptr = (*patch)[cell];
        cell_ptr->get_mg_dof_indices(local_dof_indices);

        if (is_ghost)
          for (unsigned int ind = 0; ind < local_dof_indices.size(); ++ind)
            patch_dofs_host[patch_id * n_patch_dofs +
                            cell * local_dof_indices.size() + ind] =
              partitioner->global_to_local(local_dof_indices[ind]);
        else
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
          auto scale = d == 0 ? n_replicate : 1;
          auto pos   = std::floor(first_center[d] / h + 1 / 3);
          patch_type_host[patch_id * dim + d] =
            (pos > 0) + (pos == (scale * Util::pow(2, level) - 2));
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

  template <int dim, int fe_degree, typename Number>
  void
  LevelVertexPatch<dim, fe_degree, Number>::reinit(
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

    n_replicate = dof_handler->get_triangulation().n_cells(0);

    auto locally_owned_dofs = dof_handler->locally_owned_mg_dofs(level);
    auto locally_relevant_dofs =
      DoFTools::extract_locally_relevant_level_dofs(*dof_handler, level);

    partitioner =
      std::make_shared<Utilities::MPI::Partitioner>(locally_owned_dofs,
                                                    locally_relevant_dofs,
                                                    MPI_COMM_WORLD);

    if (use_coloring)
      n_colors = regular_vpatch_size;
    else
      n_colors = 1;

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

    graph_ptr_raw.clear();
    graph_ptr_raw_ghost.clear();
    graph_ptr_raw.resize(1);
    graph_ptr_raw_ghost.resize(1);
    for (auto patch = cell_collections.begin(); patch != cell_collections.end();
         ++patch)
      {
        bool is_local = true;
        for (auto &cell : *patch)
          if (cell->is_locally_owned_on_level() == false)
            {
              is_local = false;
              break;
            }

        if (is_local)
          graph_ptr_raw[0].push_back(patch);
        else
          graph_ptr_raw_ghost[0].push_back(patch);
      }

    // coloring
    graph_ptr_colored.clear();
    graph_ptr_colored_ghost.clear();
    graph_ptr_colored.resize(regular_vpatch_size);
    graph_ptr_colored_ghost.resize(regular_vpatch_size);
    for (auto patch = cell_collections.begin(); patch != cell_collections.end();
         ++patch)
      {
        bool is_local = true;
        for (auto &cell : *patch)
          if (cell->is_locally_owned_on_level() == false)
            {
              is_local = false;
              break;
            }

        auto first_cell = (*patch)[0];

        if (is_local)
          graph_ptr_colored[first_cell->parent()->child_iterator_to_index(
                              first_cell)]
            .push_back(patch);
        else
          graph_ptr_colored_ghost[first_cell->parent()->child_iterator_to_index(
                                    first_cell)]
            .push_back(patch);
      }


    setup_color_arrays(n_colors);

    for (unsigned int i = 0; i < regular_vpatch_size; ++i)
      {
        auto n_patches      = graph_ptr_colored[i].size();
        n_patches_smooth[i] = n_patches;

        patch_type_host.clear();
        patch_id_host.clear();
        first_dof_host.clear();
        patch_id_host.resize(n_patches);
        patch_type_host.resize(n_patches * dim);
        first_dof_host.resize(n_patches * regular_vpatch_size);

        auto patch     = graph_ptr_colored[i].begin(),
             end_patch = graph_ptr_colored[i].end();
        for (unsigned int p_id = 0; patch != end_patch; ++patch, ++p_id)
          get_patch_data(*patch, p_id, false);

        alloc_arrays(&first_dof_smooth[i], n_patches * regular_vpatch_size);

        cudaError_t error_code =
          cudaMemcpy(first_dof_smooth[i],
                     first_dof_host.data(),
                     regular_vpatch_size * n_patches * sizeof(unsigned int),
                     cudaMemcpyHostToDevice);
        AssertCuda(error_code);
      }

    for (unsigned int i = 0; i < regular_vpatch_size; ++i)
      {
        auto n_patches            = graph_ptr_colored_ghost[i].size();
        n_patches_smooth_ghost[i] = n_patches;

        patch_type_host.clear();
        patch_id_host.clear();
        patch_dofs_host.clear();
        patch_id_host.resize(n_patches);
        patch_type_host.resize(n_patches * dim);
        patch_dofs_host.resize(n_patches * n_patch_dofs);

        auto patch     = graph_ptr_colored_ghost[i].begin(),
             end_patch = graph_ptr_colored_ghost[i].end();
        for (unsigned int p_id = 0; patch != end_patch; ++patch, ++p_id)
          get_patch_data(*patch, p_id, true);

        alloc_arrays(&patch_dofs_smooth[i], n_patches * n_patch_dofs);

        cudaError_t error_code =
          cudaMemcpy(patch_dofs_smooth[i],
                     patch_dofs_host.data(),
                     n_patch_dofs * n_patches * sizeof(unsigned int),
                     cudaMemcpyHostToDevice);
        AssertCuda(error_code);
      }

    std::vector<std::vector<PatchIterator>> tmp_ptr;
    tmp_ptr = use_coloring ? graph_ptr_colored : graph_ptr_raw;

    ordering_to_type.clear();
    ordering_types = 0;
    for (unsigned int i = 0; i < n_colors; ++i)
      {
        auto n_patches       = tmp_ptr[i].size();
        n_patches_laplace[i] = n_patches;

        patch_type_host.clear();
        patch_id_host.clear();
        first_dof_host.clear();
        patch_id_host.resize(n_patches);
        patch_type_host.resize(n_patches * dim);
        first_dof_host.resize(n_patches * regular_vpatch_size);

        auto patch = tmp_ptr[i].begin(), end_patch = tmp_ptr[i].end();
        for (unsigned int p_id = 0; patch != end_patch; ++patch, ++p_id)
          get_patch_data(*patch, p_id, false);

        // alloc_and_copy_arrays(i);
        alloc_arrays(&first_dof_laplace[i], n_patches * regular_vpatch_size);
        alloc_arrays(&patch_id[i], n_patches);
        alloc_arrays(&patch_type[i], n_patches * dim);

        cudaError_t error_code =
          cudaMemcpy(first_dof_laplace[i],
                     first_dof_host.data(),
                     regular_vpatch_size * n_patches * sizeof(unsigned int),
                     cudaMemcpyHostToDevice);
        AssertCuda(error_code);

        error_code = cudaMemcpy(patch_type[i],
                                patch_type_host.data(),
                                dim * n_patches * sizeof(unsigned int),
                                cudaMemcpyHostToDevice);
        AssertCuda(error_code);
      }

    for (unsigned int i = 0; i < n_colors; ++i)
      {
        auto n_patches             = graph_ptr_raw_ghost[i].size();
        n_patches_laplace_ghost[i] = n_patches;

        patch_type_host.clear();
        patch_id_host.clear();
        patch_dofs_host.clear();
        patch_id_host.resize(n_patches);
        patch_type_host.resize(n_patches * dim);
        patch_dofs_host.resize(n_patches * n_patch_dofs);

        auto patch     = graph_ptr_raw_ghost[i].begin(),
             end_patch = graph_ptr_raw_ghost[i].end();
        for (unsigned int p_id = 0; patch != end_patch; ++patch, ++p_id)
          get_patch_data(*patch, p_id, true);

        // alloc_and_copy_arrays(i);
        alloc_arrays(&patch_type_ghost[i], n_patches * dim);
        alloc_arrays(&patch_dofs_laplace[i], n_patches * n_patch_dofs);

        cudaError_t error_code =
          cudaMemcpy(patch_type_ghost[i],
                     patch_type_host.data(),
                     dim * n_patches * sizeof(unsigned int),
                     cudaMemcpyHostToDevice);
        AssertCuda(error_code);

        error_code = cudaMemcpy(patch_dofs_laplace[i],
                                patch_dofs_host.data(),
                                n_patch_dofs * n_patches * sizeof(unsigned int),
                                cudaMemcpyHostToDevice);
        AssertCuda(error_code);
      }

    setup_configuration(n_colors);

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

    DoFMapping<dim, fe_degree> dm;

    auto hl_dg          = dm.get_h_to_l_dg_normal();
    auto hl_dg_interior = dm.get_h_to_l_dg_normal_interior();

    alloc_arrays(&l_to_h, hl_dg_interior.size());
    alloc_arrays(&h_to_l, hl_dg.size());

    cudaError_t error_code =
      cudaMemcpy(l_to_h,
                 hl_dg_interior.data(),
                 hl_dg_interior.size() * sizeof(unsigned int),
                 cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    error_code = cudaMemcpy(h_to_l,
                            hl_dg.data(),
                            hl_dg.size() * sizeof(unsigned int),
                            cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    std::array<unsigned int, 8 * 8 * 8> numbering2_host;
    for (unsigned int i = 0; i < 8; ++i)
      for (unsigned int j = 0; j < 8; ++j)
        for (unsigned int k = 0; k < 8; ++k)
          numbering2_host[i * 8 * 8 + j * 8 + k] = j * 8 * 8 + i * 8 + k;

    error_code =
      cudaMemcpyToSymbol(numbering2,
                         numbering2_host.data(),
                         numbering2_host.size() * sizeof(unsigned int),
                         0,
                         cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    constexpr unsigned n_dofs_2d = n_dofs_1d * n_dofs_1d;

    alloc_arrays(&eigenvalues, n_dofs_1d * 2);
    alloc_arrays(&eigenvectors, n_dofs_2d * 2);
    alloc_arrays(&smooth_mass_1d, n_dofs_2d);
    alloc_arrays(&smooth_stiff_1d, n_dofs_2d);
    alloc_arrays(&laplace_mass_1d, n_dofs_2d * 3);
    alloc_arrays(&laplace_stiff_1d, n_dofs_2d * 3);

    reinit_tensor_product_laplace();
    reinit_tensor_product_smoother();

    AssertCuda(cudaStreamCreate(&stream));
    AssertCuda(cudaStreamCreate(&stream_g));

    // ghost
    auto ghost_indices = partitioner->ghost_indices();
    auto local_range   = partitioner->local_range();
    n_ghost_indices    = ghost_indices.n_elements();

    local_range_start = local_range.first;
    local_range_end   = local_range.second;

    auto *ghost_indices_host = new unsigned int[n_ghost_indices];
    for (unsigned int i = 0; i < n_ghost_indices; ++i)
      ghost_indices_host[i] = ghost_indices.nth_index_in_set(i);

    alloc_arrays(&ghost_indices_dev, n_ghost_indices);
    error_code = cudaMemcpy(ghost_indices_dev,
                            ghost_indices_host,
                            n_ghost_indices * sizeof(unsigned int),
                            cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    solution_ghosted = std::make_shared<
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>>();
    solution_ghosted->reinit(partitioner);

    ordering_to_type.clear();
    patch_id_host.clear();
    patch_type_host.clear();
    first_dof_host.clear();
    h_to_l_host.clear();
    l_to_h_host.clear();

    patch_id_host.shrink_to_fit();
    patch_type_host.shrink_to_fit();
    first_dof_host.shrink_to_fit();
    h_to_l_host.shrink_to_fit();
    l_to_h_host.shrink_to_fit();

    delete[] ghost_indices_host;
  }

  template <int dim, int fe_degree, typename Number>
  LevelVertexPatch<dim, fe_degree, Number>::Data
  LevelVertexPatch<dim, fe_degree, Number>::get_laplace_data(
    unsigned int color) const
  {
    Data data_copy;

    data_copy.n_patches        = n_patches_laplace[color];
    data_copy.patch_per_block  = patch_per_block;
    data_copy.first_dof        = first_dof_laplace[color];
    data_copy.patch_id         = patch_id[color];
    data_copy.patch_type       = patch_type[color];
    data_copy.l_to_h           = l_to_h;
    data_copy.h_to_l           = h_to_l;
    data_copy.laplace_mass_1d  = laplace_mass_1d;
    data_copy.laplace_stiff_1d = laplace_stiff_1d;

    data_copy.n_ghost_indices   = n_ghost_indices;
    data_copy.local_range_start = local_range_start;
    data_copy.local_range_end   = local_range_end;
    data_copy.ghost_indices     = ghost_indices_dev;

    return data_copy;
  }

  template <int dim, int fe_degree, typename Number>
  LevelVertexPatch<dim, fe_degree, Number>::Data
  LevelVertexPatch<dim, fe_degree, Number>::get_laplace_data_ghost(
    unsigned int color) const
  {
    Data data_copy;

    data_copy.n_patches        = n_patches_laplace_ghost[color];
    data_copy.patch_per_block  = patch_per_block;
    data_copy.patch_type       = patch_type_ghost[color];
    data_copy.l_to_h           = l_to_h;
    data_copy.h_to_l           = h_to_l;
    data_copy.laplace_mass_1d  = laplace_mass_1d;
    data_copy.laplace_stiff_1d = laplace_stiff_1d;

    data_copy.n_ghost_indices   = n_ghost_indices;
    data_copy.local_range_start = local_range_start;
    data_copy.local_range_end   = local_range_end;
    data_copy.ghost_indices     = ghost_indices_dev;

    data_copy.patch_dofs = patch_dofs_laplace[color];

    return data_copy;
  }

  template <int dim, int fe_degree, typename Number>
  LevelVertexPatch<dim, fe_degree, Number>::Data
  LevelVertexPatch<dim, fe_degree, Number>::get_smooth_data(
    unsigned int color) const
  {
    Data data_copy;

    data_copy.n_patches       = n_patches_smooth[color];
    data_copy.patch_per_block = patch_per_block;
    data_copy.relaxation      = relaxation;
    data_copy.first_dof       = first_dof_smooth[color];
    data_copy.l_to_h          = l_to_h;
    data_copy.h_to_l          = h_to_l;
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
  LevelVertexPatch<dim, fe_degree, Number>::Data
  LevelVertexPatch<dim, fe_degree, Number>::get_smooth_data_ghost(
    unsigned int color) const
  {
    Data data_copy;

    data_copy.n_patches       = n_patches_smooth_ghost[color];
    data_copy.patch_per_block = patch_per_block;
    data_copy.relaxation      = relaxation;
    data_copy.l_to_h          = l_to_h;
    data_copy.h_to_l          = h_to_l;
    data_copy.eigenvalues     = eigenvalues;
    data_copy.eigenvectors    = eigenvectors;
    data_copy.smooth_mass_1d  = smooth_mass_1d;
    data_copy.smooth_stiff_1d = smooth_stiff_1d;

    data_copy.n_ghost_indices   = n_ghost_indices;
    data_copy.local_range_start = local_range_start;
    data_copy.local_range_end   = local_range_end;
    data_copy.ghost_indices     = ghost_indices_dev;

    data_copy.patch_dofs = patch_dofs_smooth[color];

    return data_copy;
  }

  template <int dim, int fe_degree, typename Number>
  template <typename Operator, typename VectorType>
  void
  LevelVertexPatch<dim, fe_degree, Number>::patch_loop(const Operator   &op,
                                                       const VectorType &src,
                                                       VectorType &dst) const
  {
    Util::adjust_ghost_range_if_necessary(src, partitioner);
    Util::adjust_ghost_range_if_necessary(dst, partitioner);

    src.update_ghost_values();

    // TODO
    for (unsigned int i = 0; i < regular_vpatch_size; ++i)
      {
        (*solution_ghosted) = 0;
        // dst.update_ghost_values();
        dst.update_ghost_values_start(0);

        op.template setup_kernel<false>(patch_per_block);

        if (n_patches_smooth[i] > 0)
          {
            op.template loop_kernel<VectorType, Data, false>(
              src,
              dst,
              *solution_ghosted,
              get_smooth_data(i),
              grid_dim_smooth[i],
              block_dim_smooth[i],
              stream);

            AssertCudaKernel();
          }

        op.template setup_kernel<true>(patch_per_block);

        dst.update_ghost_values_finish();

        if (n_patches_smooth_ghost[i] > 0)
          {
            op.template loop_kernel<VectorType, Data, true>(
              src,
              dst,
              *solution_ghosted,
              get_smooth_data_ghost(i),
              grid_dim_smooth_ghost[i],
              block_dim_smooth[i],
              stream_g);

            AssertCudaKernel();
          }

        solution_ghosted->compress(VectorOperation::add);
        dst.add(1., *solution_ghosted);
      }
  }

  template <int dim, int fe_degree, typename Number>
  template <typename Operator, typename VectorType>
  void
  LevelVertexPatch<dim, fe_degree, Number>::cell_loop(const Operator   &op,
                                                      const VectorType &src,
                                                      VectorType &dst) const
  {
    Util::adjust_ghost_range_if_necessary(src, partitioner);
    Util::adjust_ghost_range_if_necessary(dst, partitioner);

    src.update_ghost_values();

    op.template setup_kernel<false>(patch_per_block);

    for (unsigned int i = 0; i < n_colors; ++i)
      if (n_patches_laplace[i] > 0)
        {
          op.template loop_kernel<VectorType, Data, false>(src,
                                                           dst,
                                                           get_laplace_data(i),
                                                           grid_dim_lapalce[i],
                                                           block_dim_laplace[i],
                                                           stream);

          AssertCudaKernel();
        }

    op.template setup_kernel<true>(patch_per_block);

    for (unsigned int i = 0; i < n_colors; ++i)
      if (n_patches_laplace_ghost[i] > 0)
        {
          op.template loop_kernel<VectorType, Data, true>(
            src,
            dst,
            get_laplace_data_ghost(i),
            grid_dim_lapalce_ghost[i],
            block_dim_laplace[i],
            stream_g);

          AssertCudaKernel();
        }
    dst.compress(VectorOperation::add);
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelVertexPatch<dim, fe_degree, Number>::reinit_tensor_product_smoother()
    const
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
        {
        }
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

    std::array<Table<2, Number>, dim> patch_mass_inv_2;
    std::array<Table<2, Number>, dim> patch_laplace_inv_2;

    for (unsigned int d = 0; d < dim; ++d)
      {
        patch_mass_inv_2[d].reinit(2 * N - 4, 2 * N - 4);
        patch_laplace_inv_2[d].reinit(2 * N - 4, 2 * N - 4);
      }

    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int i = 0; i < 2 * N - 4; ++i)
        for (unsigned int j = 0; j < 2 * N - 4; ++j)
          {
            patch_mass_inv_2[d](i, j)    = patch_mass[d](i + 2, j + 2);
            patch_laplace_inv_2[d](i, j) = patch_laplace[d](i + 2, j + 2);
          }

    // eigenvalue, eigenvector
    TensorProductData<dim, fe_degree, Number> tensor_product;
    tensor_product.reinit(patch_mass_inv, patch_laplace_inv);

    TensorProductData<dim, fe_degree, Number> tensor_product_2;
    tensor_product_2.reinit(patch_mass_inv_2, patch_laplace_inv_2);

    std::array<AlignedVector<Number>, dim> eigenval;
    std::array<Table<2, Number>, dim>      eigenvec;
    tensor_product.get_eigenvalues(eigenval);
    tensor_product.get_eigenvectors(eigenvec);

    std::array<AlignedVector<Number>, dim> eigenval_2;
    std::array<Table<2, Number>, dim>      eigenvec_2;
    tensor_product_2.get_eigenvalues(eigenval_2);
    tensor_product_2.get_eigenvectors(eigenvec_2);

    constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;

    auto *mass    = new Number[n_dofs_1d * n_dofs_1d * dim];
    auto *laplace = new Number[n_dofs_1d * n_dofs_1d * dim];
    auto *values  = new Number[n_dofs_1d * n_dofs_1d * dim * 2];
    auto *vectors = new Number[n_dofs_1d * n_dofs_1d * dim * 2];

    for (int d = 0; d < 1; ++d)
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

    std::transform(eigenval_2[0].begin(),
                   eigenval_2[0].end(),
                   &values[n_dofs_1d],
                   [](const Number m) -> Number { return m; });

    std::transform(eigenvec_2[0].begin(),
                   eigenvec_2[0].end(),
                   &vectors[n_dofs_1d * n_dofs_1d],
                   [](const Number m) -> Number { return m; });

    cudaError_t error_code = cudaMemcpy(eigenvalues,
                                        values,
                                        2 * n_dofs_1d * sizeof(Number),
                                        cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    error_code = cudaMemcpy(eigenvectors,
                            vectors,
                            2 * n_dofs_1d * n_dofs_1d * sizeof(Number),
                            cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    error_code = cudaMemcpy(smooth_mass_1d,
                            mass,
                            n_dofs_1d * n_dofs_1d * sizeof(Number),
                            cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    error_code = cudaMemcpy(smooth_stiff_1d,
                            laplace,
                            n_dofs_1d * n_dofs_1d * sizeof(Number),
                            cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    delete[] mass;
    delete[] laplace;
    delete[] values;
    delete[] vectors;
  }


  template <int dim, int fe_degree, typename Number>
  void
  LevelVertexPatch<dim, fe_degree, Number>::reinit_tensor_product_laplace()
    const
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


    cudaError_t error_code = cudaMemcpy(laplace_mass_1d,
                                        mass,
                                        3 * n_dofs_2d * sizeof(Number),
                                        cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    error_code = cudaMemcpy(laplace_stiff_1d,
                            laplace,
                            3 * n_dofs_2d * sizeof(Number),
                            cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    delete[] mass;
    delete[] laplace;
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelVertexPatch<dim, fe_degree, Number>::setup_color_arrays(
    const unsigned int n_colors)
  {
    this->n_patches_laplace.resize(n_colors);
    this->grid_dim_lapalce.resize(n_colors);
    this->block_dim_laplace.resize(n_colors);
    this->first_dof_laplace.resize(n_colors);
    this->patch_id.resize(n_colors);
    this->patch_type.resize(n_colors);

    this->n_patches_smooth.resize(regular_vpatch_size);
    this->grid_dim_smooth.resize(regular_vpatch_size);
    this->block_dim_smooth.resize(regular_vpatch_size);
    this->first_dof_smooth.resize(regular_vpatch_size);

    this->n_patches_laplace_ghost.resize(n_colors);
    this->grid_dim_lapalce_ghost.resize(n_colors);
    this->patch_type_ghost.resize(n_colors);

    this->n_patches_smooth_ghost.resize(regular_vpatch_size);
    this->grid_dim_smooth_ghost.resize(regular_vpatch_size);

    this->patch_dofs_laplace.resize(n_colors);
    this->patch_dofs_smooth.resize(regular_vpatch_size);
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelVertexPatch<dim, fe_degree, Number>::setup_configuration(
    const unsigned int n_colors)
  {
    constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;

    for (unsigned int i = 0; i < n_colors; ++i)
      {
        auto   n_patches      = n_patches_laplace[i];
        double apply_n_blocks = std::ceil(static_cast<double>(n_patches) /
                                          static_cast<double>(patch_per_block));

        grid_dim_lapalce[i]  = dim3(apply_n_blocks);
        block_dim_laplace[i] = dim3(patch_per_block * n_dofs_1d, n_dofs_1d);

        n_patches      = n_patches_laplace_ghost[i];
        apply_n_blocks = std::ceil(static_cast<double>(n_patches) /
                                   static_cast<double>(patch_per_block));

        grid_dim_lapalce_ghost[i] = dim3(apply_n_blocks);
      }

    for (unsigned int i = 0; i < regular_vpatch_size; ++i)
      {
        auto   n_patches      = n_patches_smooth[i];
        double apply_n_blocks = std::ceil(static_cast<double>(n_patches) /
                                          static_cast<double>(patch_per_block));

        grid_dim_smooth[i]  = dim3(apply_n_blocks);
        block_dim_smooth[i] = dim3(patch_per_block * n_dofs_1d, n_dofs_1d);

        n_patches      = n_patches_smooth_ghost[i];
        apply_n_blocks = std::ceil(static_cast<double>(n_patches) /
                                   static_cast<double>(patch_per_block));

        grid_dim_smooth_ghost[i] = dim3(apply_n_blocks);
      }
  }

  template <int dim, int fe_degree, typename Number>
  template <typename Number1>
  void
  LevelVertexPatch<dim, fe_degree, Number>::alloc_arrays(Number1 **array_device,
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