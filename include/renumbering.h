/**
 * @file renumber.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief Mapping between lexicographic and hierarchic numbering
 * @version 1.0
 * @date 2022-12-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef RENUMBER_H
#define RENUMBER_H

#include "utilities.cuh"

using namespace dealii;

namespace PSMF
{
  template <int dim, int fe_degree>
  class DoFMapping
  {
  public:
    using value_type = unsigned int;

    DoFMapping()
    {
      FE_DGQLegendre<dim> fe_p(fe_degree);

      Triangulation<dim> triangulation(
        Triangulation<dim>::limit_level_difference_at_vertices);
      GridGenerator::hyper_cube(triangulation, 0., 1.);
      triangulation.refine_global(1);

      DoFHandler<dim> dof_handler_p(triangulation);
      dof_handler_p.distribute_dofs(fe_p);

      // DG
      {
        const value_type        dofs_per_cell = fe_p.n_dofs_per_cell();
        std::vector<value_type> local_dof_indices(dofs_per_cell);

        std::array<std::vector<value_type>, 1 << dim> cell_dofs;

        unsigned int c = 0;
        for (const auto &cell : dof_handler_p.active_cell_iterators())
          {
            cell->get_dof_indices(local_dof_indices);

            cell_dofs[c].resize(dofs_per_cell);
            for (auto i = 0U; i < dofs_per_cell; ++i)
              cell_dofs[c][i] = local_dof_indices[i];

            c++;
          }

        h_to_l_dg_normal =
          form_patch_dg_lexicographic_numbering_normal(cell_dofs);
        h_to_l_dg_normal_interior =
          get_patch_dg_lexicographic_numbering_interior(h_to_l_dg_normal);
      }
    }

    std::vector<value_type>
    get_h_to_l_dg_normal()
    {
      return h_to_l_dg_normal;
    }
    std::vector<value_type>
    get_h_to_l_dg_normal_interior()
    {
      return h_to_l_dg_normal_interior;
    }

  private:
    std::vector<value_type>
    form_patch_dg_lexicographic_numbering_normal(
      std::array<std::vector<value_type>, 1 << dim> &cell_dofs)
    {
      std::vector<value_type> local_dof_indices;

      const unsigned int layer = dim == 2 ? 1 : fe_degree + 1;

      for (auto z = 0U; z < dim - 1; ++z)
        for (auto l = 0U; l < layer; ++l)
          for (auto row = 0U; row < 2; ++row)
            for (auto i = 0U; i < fe_degree + 1; ++i)
              for (auto col = 0U; col < 2; ++col)
                for (auto k = 0U; k < fe_degree + 1; ++k)
                  {
                    const unsigned int cell = z * 4 + row * 2 + col;

                    local_dof_indices.push_back(
                      cell_dofs[cell][l * (fe_degree + 1) * (fe_degree + 1) +
                                      i * (fe_degree + 1) + k]);
                  }

      return local_dof_indices;
    }

    std::vector<value_type>
    get_patch_dg_lexicographic_numbering_interior(
      std::vector<value_type> &patch_numbering)
    {
      std::vector<value_type> local_dof_indices;

      constexpr unsigned int n_z = dim == 2 ? 1 : 2 * fe_degree;

      for (auto i = 0U; i < n_z; ++i)
        for (auto j = 0U; j < 2 * fe_degree; ++j)
          for (auto k = 0U; k < 2 * fe_degree; ++k)
            {
              local_dof_indices.push_back(
                patch_numbering[(i + dim - 2) * (2 * fe_degree + 2) *
                                  (2 * fe_degree + 2) +
                                (j + 1) * (2 * fe_degree + 2) + k + 1]);
            }

      return local_dof_indices;
    }


    std::vector<value_type>
    reverse_numbering(std::vector<value_type> &l_numbering)
    {
      auto sortedVector = l_numbering;
      std::sort(sortedVector.begin(), sortedVector.end());

      std::vector<value_type> local_dof_indices(l_numbering.size());
      for (auto i = 0U; i < l_numbering.size(); ++i)
        {
          auto it =
            std::find(l_numbering.begin(), l_numbering.end(), sortedVector[i]);

          local_dof_indices[i] = std::distance(l_numbering.begin(), it);
        }
      return local_dof_indices;
    }

    std::vector<value_type> h_to_l_dg_normal;
    std::vector<value_type> h_to_l_dg_normal_interior;
  };

} // namespace PSMF


#endif // RENUMBER_H