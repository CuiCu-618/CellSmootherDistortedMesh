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

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_raviart_thomas_new.h>

#include "utilities.cuh"

using namespace dealii;

namespace PSMF
{
  template <int dim, int fe_degree = -1>
  class DoFMapping
  {
  public:
    using value_type = unsigned int;

    DoFMapping()
    {
      FE_RaviartThomas_new<dim> fe_v(fe_degree);
      FE_DGQLegendre<dim>       fe_p(fe_degree);

      Triangulation<dim> triangulation(
        Triangulation<dim>::limit_level_difference_at_vertices);
      GridGenerator::hyper_cube(triangulation, 0., 1.);
      triangulation.refine_global(1);

      DoFHandler<dim> dof_handler_v(triangulation);
      dof_handler_v.distribute_dofs(fe_v);

      DoFHandler<dim> dof_handler_p(triangulation);
      dof_handler_p.distribute_dofs(fe_p);

      // RT
      {
        auto cell_rt_lnumbering =
          get_cell_rt_lexicographic_numbering(fe_degree + 1, fe_degree);
        auto cell_rt_lnumbering_x =
          get_cell_rt_lexicographic_numbering_xfast(fe_degree + 1, fe_degree);

        lexicographic_numbering = cell_rt_lnumbering;

        const value_type        dofs_per_cell = fe_v.n_dofs_per_cell();
        std::vector<value_type> local_dof_indices(dofs_per_cell);

        std::array<std::vector<value_type>, 1 << dim> cell_dofs;
        std::array<std::vector<value_type>, 1 << dim> cell_dofs_x;

        value_type c = 0;
        for (const auto &cell : dof_handler_v.active_cell_iterators())
          {
            cell->get_dof_indices(local_dof_indices);

            cell_dofs[c].resize(dofs_per_cell);
            cell_dofs_x[c].resize(dofs_per_cell);
            for (auto i = 0U; i < dofs_per_cell; ++i)
              {
                cell_dofs[c][i]   = local_dof_indices[cell_rt_lnumbering[i]];
                cell_dofs_x[c][i] = local_dof_indices[cell_rt_lnumbering_x[i]];
              }

            c++;
          }

        h_to_l_rt = form_patch_rt_lexicographic_numbering(cell_dofs);
        h_to_l_rt_interior =
          get_patch_rt_lexicographic_numbering_interior(h_to_l_rt);

        h_to_l_rt_x = form_patch_rt_lexicographic_numbering_xfast(cell_dofs_x);
        h_to_l_rt_interior_x =
          get_patch_rt_lexicographic_numbering_interior_xfast(h_to_l_rt_x);

        l_to_h_rt          = reverse_numbering(h_to_l_rt);
        l_to_h_rt_interior = reverse_numbering(h_to_l_rt_interior);
      }

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
        h_to_l_dg_tangent =
          form_patch_dg_lexicographic_numbering_tangent(cell_dofs);

        if (dim == 3)
          h_to_l_dg_z = form_patch_dg_lexicographic_numbering_z(cell_dofs);

        h_to_l_dg_normal_interior =
          get_patch_dg_lexicographic_numbering_interior(h_to_l_dg_normal);
        h_to_l_dg_tangent_interior =
          get_patch_dg_lexicographic_numbering_interior(h_to_l_dg_tangent);

        l_to_h_dg_normal  = reverse_numbering(h_to_l_dg_normal);
        l_to_h_dg_tangent = reverse_numbering(h_to_l_dg_tangent);

        if (dim == 3)
          l_to_h_dg_z = reverse_numbering(h_to_l_dg_z);

        l_to_h_dg_normal_interior =
          reverse_numbering(h_to_l_dg_normal_interior);
        l_to_h_dg_tangent_interior =
          reverse_numbering(h_to_l_dg_tangent_interior);
      }

      // first dof rt
      {
        constexpr int n_cells   = 1 << dim;
        constexpr int face_dofs = Util::pow(fe_degree + 1, dim - 1);
        constexpr int quad_dofs =
          dim * Util::pow(fe_degree + 1, dim - 1) * fe_degree;

        std::vector<int> cell_faces;
        if (dim == 2)
          cell_faces = {{4, 3, 3, 2}};
        else
          cell_faces = {{6, 5, 5, 4, 5, 4, 4, 3}};

        std::vector<value_type> cell_face_dofs;
        std::vector<value_type> cell_dofs;

        for (auto c = 0; c < n_cells; ++c)
          {
            cell_face_dofs.push_back(cell_faces[c] * face_dofs);
            cell_dofs.push_back(cell_face_dofs[c] + quad_dofs);
          }

        int start = 0;
        for (int c = 0; c < n_cells; ++c)
          {
            for (int f = 0; f < cell_faces[c]; ++f)
              {
                first_dofs_rt.push_back(start);
                start += face_dofs;
              }
            first_dofs_rt.push_back(start);
            start += quad_dofs;
          }

        int base   = -1;
        int offset = -1;

        for (int tid = 0; tid < h_to_l_rt.size(); ++tid)
          {
            for (int c = 0; c < n_cells; ++c)
              {
                int patch_dof = 0;
                for (int subc = 0; subc < c + 1; ++subc)
                  patch_dof += cell_dofs[subc];

                if (tid < patch_dof) // cell
                  {
                    int local_tid = tid - patch_dof + cell_dofs[c];

                    if (local_tid >= cell_face_dofs[c]) // quad dof
                      {
                        int shift = -1;
                        for (int subc = 0; subc <= c; ++subc)
                          shift += (1 + cell_faces[subc]);
                        base   = shift;
                        offset = local_tid - cell_face_dofs[c];

                        goto exitLoop;
                      }

                    for (int f = 0; f < cell_faces[c]; ++f)
                      if (local_tid < (f + 1) * face_dofs) // face dof
                        {
                          int shift = 0;
                          for (int subc = 0; subc < c; ++subc)
                            shift += (1 + cell_faces[subc]);

                          base   = shift + f;
                          offset = local_tid - f * face_dofs;
                          goto exitLoop;
                        }
                  }
              }
          exitLoop:
            base_dof_rt.push_back(base);
            dof_offset_rt.push_back(offset);
          }
      }

      // first dof dg
      {
        constexpr int n_cells   = 1 << dim;
        constexpr int quad_dofs = Util::pow(fe_degree + 1, dim);

        for (int c = 0; c < n_cells; ++c)
          for (int ind = 0; ind < quad_dofs; ++ind)
            {
              base_dof_dg.push_back(c);
              dof_offset_dg.push_back(ind);
            }
      }
    }

    DoFMapping(const unsigned int degree)
    {
      lexicographic_numbering =
        get_cell_rt_lexicographic_numbering(degree + 1, degree);

      // first dof rt
      {
        int n_cells   = 1 << dim;
        int face_dofs = Util::pow(degree + 1, dim - 1);
        int quad_dofs = dim * Util::pow(degree + 1, dim - 1) * degree;
        int n_patch_dofs =
          dim * Util::pow(2 * degree + 2, dim - 1) * (2 * degree + 3);

        std::vector<int> cell_faces;
        if (dim == 2)
          cell_faces = {{4, 3, 3, 2}};
        else
          cell_faces = {{6, 5, 5, 4, 5, 4, 4, 3}};

        std::vector<value_type> cell_face_dofs;
        std::vector<value_type> cell_dofs;

        for (auto c = 0; c < n_cells; ++c)
          {
            cell_face_dofs.push_back(cell_faces[c] * face_dofs);
            cell_dofs.push_back(cell_face_dofs[c] + quad_dofs);
          }

        int start = 0;
        for (int c = 0; c < n_cells; ++c)
          {
            for (int f = 0; f < cell_faces[c]; ++f)
              {
                first_dofs_rt.push_back(start);
                start += face_dofs;
              }
            first_dofs_rt.push_back(start);
            start += quad_dofs;

            if (c == 0)
              {
                first_dofs_cell = first_dofs_rt;
                first_dofs_cell.push_back(start);
              }
          }

        first_dofs = first_dofs_rt;
        for (int c = 0; c < n_cells; ++c)
          {
            first_dofs.push_back(start);
            start += Util::pow(degree + 1, dim);
          }

        int base   = -1;
        int offset = -1;

        for (int tid = 0; tid < n_patch_dofs; ++tid)
          {
            for (int c = 0; c < n_cells; ++c)
              {
                int patch_dof = 0;
                for (int subc = 0; subc < c + 1; ++subc)
                  patch_dof += cell_dofs[subc];

                if (tid < patch_dof) // cell
                  {
                    int local_tid = tid - patch_dof + cell_dofs[c];

                    if (local_tid >= cell_face_dofs[c]) // quad dof
                      {
                        int shift = -1;
                        for (int subc = 0; subc <= c; ++subc)
                          shift += (1 + cell_faces[subc]);
                        base   = shift;
                        offset = local_tid - cell_face_dofs[c];

                        goto exitLoop;
                      }

                    for (int f = 0; f < cell_faces[c]; ++f)
                      if (local_tid < (f + 1) * face_dofs) // face dof
                        {
                          int shift = 0;
                          for (int subc = 0; subc < c; ++subc)
                            shift += (1 + cell_faces[subc]);

                          base   = shift + f;
                          offset = local_tid - f * face_dofs;
                          goto exitLoop;
                        }
                  }
              }
          exitLoop:
            base_dof_rt.push_back(base);
            dof_offset_rt.push_back(offset);

            if (tid == cell_faces[0] * face_dofs + quad_dofs - 1)
              {
                base_dof_cell   = base_dof_rt;
                dof_offset_cell = dof_offset_rt;
              }
          }
      }

      // first dof dg
      {
        int n_cells   = 1 << dim;
        int quad_dofs = Util::pow(degree + 1, dim);

        for (int c = 0; c < n_cells; ++c)
          {
            for (int ind = 0; ind < quad_dofs; ++ind)
              {
                base_dof_dg.push_back(c);
                dof_offset_dg.push_back(ind);
              }

            if (c == 0)
              {
                auto n_dof_cell_rt = 2 * dim + 1;
                for (auto ii = 0U; ii < base_dof_dg.size(); ++ii)
                  base_dof_cell.push_back(base_dof_dg[0] + n_dof_cell_rt);

                dof_offset_cell.insert(dof_offset_cell.end(),
                                       dof_offset_dg.begin(),
                                       dof_offset_dg.end());
              }
          }
      }

      base_dof = base_dof_rt;
      for (auto ii = 0U; ii < base_dof_dg.size(); ++ii)
        base_dof.push_back(base_dof_dg[ii] + first_dofs_rt.size());

      dof_offset = dof_offset_rt;
      dof_offset.insert(dof_offset.end(),
                        dof_offset_dg.begin(),
                        dof_offset_dg.end());
    }

    std::vector<value_type>
    get_lexicographic_numbering()
    {
      return lexicographic_numbering;
    }
    std::vector<value_type>
    get_h_to_l_rt()
    {
      return h_to_l_rt;
    }
    std::vector<value_type>
    get_h_to_l_rt_interior()
    {
      return h_to_l_rt_interior;
    }
    std::vector<value_type>
    get_h_to_l_rt_x()
    {
      return h_to_l_rt_x;
    }
    std::vector<value_type>
    get_h_to_l_rt_interior_x()
    {
      return h_to_l_rt_interior_x;
    }
    std::vector<value_type>
    get_h_to_l_dg_normal()
    {
      return h_to_l_dg_normal;
    }
    std::vector<value_type>
    get_h_to_l_dg_tangent()
    {
      return h_to_l_dg_tangent;
    }
    std::vector<value_type>
    get_h_to_l_dg_z()
    {
      return h_to_l_dg_z;
    }
    std::vector<value_type>
    get_h_to_l_dg_normal_interior()
    {
      return h_to_l_dg_normal_interior;
    }
    std::vector<value_type>
    get_h_to_l_dg_tangent_interior()
    {
      return h_to_l_dg_tangent_interior;
    }
    std::vector<value_type>
    get_l_to_h_rt()
    {
      return l_to_h_rt;
    }
    std::vector<value_type>
    get_l_to_h_rt_interior()
    {
      return l_to_h_rt_interior;
    }
    std::vector<value_type>
    get_l_to_h_dg_normal()
    {
      return l_to_h_dg_normal;
    }
    std::vector<value_type>
    get_l_to_h_dg_tangent()
    {
      return l_to_h_dg_tangent;
    }
    std::vector<value_type>
    get_l_to_h_dg_z()
    {
      return l_to_h_dg_z;
    }
    std::vector<value_type>
    get_l_to_h_dg_normal_interior()
    {
      return l_to_h_dg_normal_interior;
    }
    std::vector<value_type>
    get_l_to_h_dg_tangent_interior()
    {
      return l_to_h_dg_tangent_interior;
    }
    std::vector<value_type>
    get_first_dofs_rt()
    {
      return first_dofs_rt;
    }
    std::vector<value_type>
    get_first_dofs()
    {
      return first_dofs;
    }
    std::vector<value_type>
    get_first_dofs_cell()
    {
      return first_dofs_cell;
    }
    std::vector<value_type>
    get_base_dof_rt()
    {
      return base_dof_rt;
    }
    std::vector<value_type>
    get_dof_offset_rt()
    {
      return dof_offset_rt;
    }
    std::vector<value_type>
    get_base_dof_dg()
    {
      return base_dof_dg;
    }
    std::vector<value_type>
    get_dof_offset_dg()
    {
      return dof_offset_dg;
    }
    std::vector<value_type>
    get_base_dof()
    {
      return base_dof;
    }
    std::vector<value_type>
    get_dof_offset()
    {
      return dof_offset;
    }
    std::vector<value_type>
    get_base_dof_cell()
    {
      return base_dof_cell;
    }
    std::vector<value_type>
    get_dof_offset_cell()
    {
      return dof_offset_cell;
    }

  private:
    std::vector<value_type>
    get_cell_rt_lexicographic_numbering(const value_type normal_degree,
                                        const value_type tangential_degree)
    {
      const unsigned int n_dofs_face =
        Utilities::pow(tangential_degree + 1, dim - 1);
      std::vector<unsigned int> lexicographic_numbering;
      // component 1
      for (unsigned int j = 0; j < n_dofs_face; ++j)
        {
          lexicographic_numbering.push_back(j);
          if (normal_degree > 1)
            for (unsigned int i = n_dofs_face * 2 * dim;
                 i < n_dofs_face * 2 * dim + normal_degree - 1;
                 ++i)
              lexicographic_numbering.push_back(i + j * (normal_degree - 1));
          lexicographic_numbering.push_back(n_dofs_face + j);
        }

      // component 2
      unsigned int layers = (dim == 3) ? tangential_degree + 1 : 1;
      for (unsigned int k = 0; k < layers; ++k)
        for (unsigned int j = 0; j < tangential_degree + 1; ++j)
          {
            unsigned int s = j + n_dofs_face * 2;

            unsigned int k_add = k * (tangential_degree + 1);

            lexicographic_numbering.push_back(s + k_add);

            if (normal_degree > 1)
              for (unsigned int i =
                     n_dofs_face * (2 * dim + (normal_degree - 1));
                   i < n_dofs_face * (2 * dim + (normal_degree - 1)) +
                         (normal_degree - 1) * (tangential_degree + 1);
                   i += tangential_degree + 1)
                {
                  lexicographic_numbering.push_back(i + j +
                                                    k_add * tangential_degree);
                }
            unsigned int e = j + n_dofs_face * 3;
            lexicographic_numbering.push_back(e + k_add);
          }

      // component 3
      if (dim == 3)
        {
          for (unsigned int j = 0; j < tangential_degree + 1; ++j)
            for (unsigned int k = 0; k < layers; ++k)
              {
                unsigned int j_add = j * (tangential_degree + 1);

                unsigned int s = k + 4 * n_dofs_face;
                lexicographic_numbering.push_back(s + j_add);

                if (normal_degree > 1)
                  {
                    for (unsigned int i = 6 * n_dofs_face +
                                          n_dofs_face * 2 * (normal_degree - 1);
                         i < 6 * n_dofs_face +
                               n_dofs_face * 2 * (normal_degree - 1) +
                               (normal_degree - 1) * n_dofs_face;
                         i += n_dofs_face)
                      lexicographic_numbering.push_back(i + k + j_add);
                  }

                unsigned int e = k + 5 * n_dofs_face;
                lexicographic_numbering.push_back(e + j_add);
              }
        }

      return lexicographic_numbering;
    }

    std::vector<value_type>
    get_cell_rt_lexicographic_numbering_xfast(
      const unsigned int normal_degree,
      const unsigned int tangential_degree)
    {
      const unsigned int n_dofs_face =
        Utilities::pow(tangential_degree + 1, dim - 1);
      std::vector<unsigned int> lexicographic_numbering;
      // component 1
      for (unsigned int j = 0; j < n_dofs_face; ++j)
        {
          lexicographic_numbering.push_back(j);
          if (normal_degree > 1)
            for (unsigned int i = n_dofs_face * 2 * dim;
                 i < n_dofs_face * 2 * dim + normal_degree - 1;
                 ++i)
              lexicographic_numbering.push_back(i + j * (normal_degree - 1));
          lexicographic_numbering.push_back(n_dofs_face + j);
        }

      // component 2
      unsigned int layers = (dim == 3) ? tangential_degree + 1 : 1;
      for (unsigned int k = 0; k < layers; ++k)
        {
          unsigned int k_add = k * (tangential_degree + 1);
          for (unsigned int j = n_dofs_face * 2;
               j < n_dofs_face * 2 + tangential_degree + 1;
               ++j)
            lexicographic_numbering.push_back(j + k_add);

          if (normal_degree > 1)
            for (unsigned int i = n_dofs_face * (2 * dim + (normal_degree - 1));
                 i < n_dofs_face * (2 * dim + (normal_degree - 1)) +
                       (normal_degree - 1) * (tangential_degree + 1);
                 ++i)
              {
                lexicographic_numbering.push_back(i +
                                                  k_add * tangential_degree);
              }
          for (unsigned int j = n_dofs_face * 3;
               j < n_dofs_face * 3 + tangential_degree + 1;
               ++j)
            lexicographic_numbering.push_back(j + k_add);
        }

      // component 3
      if (dim == 3)
        {
          for (unsigned int i = 4 * n_dofs_face; i < 5 * n_dofs_face; ++i)
            lexicographic_numbering.push_back(i);
          if (normal_degree > 1)
            for (unsigned int i =
                   6 * n_dofs_face + n_dofs_face * 2 * (normal_degree - 1);
                 i < 6 * n_dofs_face + n_dofs_face * 3 * (normal_degree - 1);
                 ++i)
              lexicographic_numbering.push_back(i);
          for (unsigned int i = 5 * n_dofs_face; i < 6 * n_dofs_face; ++i)
            lexicographic_numbering.push_back(i);
        }

      return lexicographic_numbering;
    }

    std::vector<value_type>
    form_patch_rt_lexicographic_numbering(
      std::array<std::vector<value_type>, 1 << dim> &cell_dofs)
    {
      std::array<std::vector<unsigned int>, dim> cell_number;

      if (dim == 2)
        {
          cell_number[0] = {{0, 1, 2, 3}};
          cell_number[1] = {{0, 2, 1, 3}};
        }
      else if (dim == 3)
        {
          cell_number[0] = {{0, 1, 2, 3, 4, 5, 6, 7}};
          cell_number[1] = {{0, 2, 1, 3, 4, 6, 5, 7}};
          cell_number[2] = {{0, 4, 1, 5, 2, 6, 3, 7}};
        }

      std::vector<value_type> local_dof_indices;

      const unsigned int layer = dim == 2 ? 1 : fe_degree + 1;

      for (auto d = 0U; d < dim; ++d)
        for (auto z = 0U; z < dim - 1; ++z)
          for (auto l = 0U; l < layer; ++l)
            for (auto row = 0U; row < 2; ++row)
              for (auto i = 0U; i < fe_degree + 1; ++i)
                for (auto col = 0U; col < 2; ++col)
                  for (auto k = 0U; k < fe_degree + 2; ++k)
                    {
                      if (k == 0 && col == 1)
                        continue;

                      const unsigned int cell =
                        cell_number[d][z * 4 + row * 2 + col];

                      local_dof_indices.push_back(
                        cell_dofs[cell][d * Util::pow(fe_degree + 1, dim - 1) *
                                          (fe_degree + 2) +
                                        l * (fe_degree + 1) * (fe_degree + 2) +
                                        i * (fe_degree + 2) + k]);
                    }

      return local_dof_indices;
    }

    std::vector<value_type>
    form_patch_rt_lexicographic_numbering_xfast(
      std::array<std::vector<value_type>, 1 << dim> &cell_dofs)
    {
      std::vector<unsigned int> cell_number;

      if (dim == 2)
        cell_number = {{0, 1, 2, 3}};
      else if (dim == 3)
        cell_number = {{0, 1, 2, 3, 4, 5, 6, 7}};

      std::vector<unsigned int> normal = {
        {fe_degree + 2, fe_degree + 1, fe_degree + 1}};
      std::vector<unsigned int> tangent = {
        {fe_degree + 1, fe_degree + 2, fe_degree + 1}};


      std::vector<value_type> local_dof_indices;

      std::vector<unsigned int> layer(3);

      if (dim == 2)
        {
          layer[0] = 1;
          layer[1] = 1;
          layer[2] = 1;
        }
      else
        {
          layer[0] = fe_degree + 1;
          layer[1] = fe_degree + 1;
          layer[2] = fe_degree + 2;
        }

      for (auto d = 0U; d < dim; ++d)
        for (auto z = 0U; z < dim - 1; ++z)
          for (auto l = 0U; l < layer[d]; ++l)
            for (auto row = 0U; row < 2; ++row)
              for (auto i = 0U; i < tangent[d]; ++i)
                for (auto col = 0U; col < 2; ++col)
                  for (auto k = 0U; k < normal[d]; ++k)
                    {
                      if ((d == 0 && k == 0 && col == 1) ||
                          (d == 1 && i == 0 && row == 1) ||
                          (d == 2 && l == 0 && z == 1))
                        continue;

                      const unsigned int cell =
                        cell_number[z * 4 + row * 2 + col];

                      local_dof_indices.push_back(
                        cell_dofs[cell][d * Util::pow(fe_degree + 1, dim - 1) *
                                          (fe_degree + 2) +
                                        l * tangent[d] * normal[d] +
                                        i * normal[d] + k]);
                    }

      return local_dof_indices;
    }

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
    form_patch_dg_lexicographic_numbering_tangent(
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
                    const unsigned int cell = z * 4 + col * 2 + row;

                    local_dof_indices.push_back(
                      cell_dofs[cell][l * (fe_degree + 1) * (fe_degree + 1) +
                                      k * (fe_degree + 1) + i]);
                  }

      return local_dof_indices;
    }

    std::vector<value_type>
    form_patch_dg_lexicographic_numbering_z(
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
                    const unsigned int cell = col * 4 + z * 2 + row;

                    local_dof_indices.push_back(
                      cell_dofs[cell][k * (fe_degree + 1) * (fe_degree + 1) +
                                      l * (fe_degree + 1) + i]);
                  }

      return local_dof_indices;
    }

    std::vector<value_type>
    get_patch_rt_lexicographic_numbering_interior(
      std::vector<value_type> &patch_numbering)
    {
      const unsigned int n_comp_dofs = patch_numbering.size() / dim;

      constexpr unsigned int n_dofs_normal = 2 * fe_degree + 3;
      constexpr unsigned int n_dofs_tang   = 2 * fe_degree + 2;

      constexpr unsigned int n_z = dim == 2 ? 1 : 2 * fe_degree + 2;

      std::vector<value_type> local_dof_indices;

      for (auto d = 0U; d < dim; ++d)
        for (auto i = 0U; i < n_z; ++i)
          for (auto j = 0U; j < 2 * fe_degree + 2; ++j)
            for (auto k = 0U; k < 2 * fe_degree + 1; ++k)
              {
                local_dof_indices.push_back(
                  patch_numbering[d * n_comp_dofs +
                                  i * n_dofs_normal * n_dofs_tang +
                                  j * n_dofs_normal + k + 1]);
              }

      return local_dof_indices;
    }

    std::vector<value_type>
    get_patch_rt_lexicographic_numbering_interior_xfast(
      std::vector<value_type> &patch_numbering)
    {
      const unsigned int n_comp_dofs = patch_numbering.size() / dim;

      std::vector<unsigned int> normal = {
        {2 * fe_degree + 1, 2 * fe_degree + 2, 2 * fe_degree + 2}};
      std::vector<unsigned int> tangent = {
        {2 * fe_degree + 2, 2 * fe_degree + 1, 2 * fe_degree + 2}};

      std::vector<unsigned int> layer(3);

      if (dim == 2)
        {
          layer[0] = 1;
          layer[1] = 1;
          layer[2] = 1;
        }
      else
        {
          layer[0] = 2 * fe_degree + 2;
          layer[1] = 2 * fe_degree + 2;
          layer[2] = 2 * fe_degree + 1;
        }

      std::vector<value_type> local_dof_indices;

      for (auto d = 0U; d < dim; ++d)
        for (auto i = 0U; i < layer[d]; ++i)
          for (auto j = 0U; j < tangent[d]; ++j)
            for (auto k = 0U; k < normal[d]; ++k)
              {
                bool si = d == 2;
                bool sj = d == 1;
                bool sk = d == 0;
                local_dof_indices.push_back(
                  patch_numbering[d * n_comp_dofs +
                                  (i + si) * (normal[d] + 2 * sk) *
                                    (tangent[d] + 2 * sj) +
                                  (j + sj) * (normal[d] + 2 * sk) + k + sk]);
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


    // cell numbering
    std::vector<value_type> lexicographic_numbering;

    // dir fast
    std::vector<value_type> h_to_l_rt;
    std::vector<value_type> h_to_l_rt_interior;

    // x fast
    std::vector<value_type> h_to_l_rt_x;
    std::vector<value_type> h_to_l_rt_interior_x;

    std::vector<value_type> h_to_l_dg_normal;
    std::vector<value_type> h_to_l_dg_tangent;
    std::vector<value_type> h_to_l_dg_z;

    std::vector<value_type> h_to_l_dg_normal_interior;
    std::vector<value_type> h_to_l_dg_tangent_interior;

    std::vector<value_type> l_to_h_rt;
    std::vector<value_type> l_to_h_rt_interior;

    std::vector<value_type> l_to_h_dg_normal;
    std::vector<value_type> l_to_h_dg_tangent;
    std::vector<value_type> l_to_h_dg_z;

    std::vector<value_type> l_to_h_dg_normal_interior;
    std::vector<value_type> l_to_h_dg_tangent_interior;

    std::vector<value_type> first_dofs_rt;
    std::vector<value_type> base_dof_rt;
    std::vector<value_type> dof_offset_rt;

    std::vector<value_type> base_dof_dg;
    std::vector<value_type> dof_offset_dg;

    std::vector<value_type> first_dofs;
    std::vector<value_type> base_dof;
    std::vector<value_type> dof_offset;

    std::vector<value_type> first_dofs_cell;
    std::vector<value_type> base_dof_cell;
    std::vector<value_type> dof_offset_cell;
  };

} // namespace PSMF


#endif // RENUMBER_H
