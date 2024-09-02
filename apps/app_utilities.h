/**
 * @file app_utilities.h
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief collection of helper functions for apps
 * @version 1.0
 * @date 2022-12-27
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef APP_UTILITIES_H
#define APP_UTILITIES_H

#include <helper_cuda.h>

#include "ct_parameter.h"
#include "git_version.h"

#define LA_MACRO(name, v1, v2, v3, v4, v5)                 \
  enum class name                                          \
  {                                                        \
    v1,                                                    \
    v2,                                                    \
    v3,                                                    \
    v4,                                                    \
    v5,                                                    \
  };                                                       \
  const char *name##Strings[] = {#v1, #v2, #v3, #v4, #v5}; \
  template <typename T>                                    \
  constexpr const char *name##ToString(T value)            \
  {                                                        \
    return name##Strings[static_cast<int>(value)];         \
  }

#define SMO_MACRO(name, v1, v2, v3, v4, v5, v6)                 \
  enum class name                                               \
  {                                                             \
    v1,                                                         \
    v2,                                                         \
    v3,                                                         \
    v4,                                                         \
    v5,                                                         \
    v6,                                                         \
  };                                                            \
  const char *name##Strings[] = {#v1, #v2, #v3, #v4, #v5, #v6}; \
  template <typename T>                                         \
  constexpr const char *name##ToString(T value)                 \
  {                                                             \
    return name##Strings[static_cast<int>(value)];              \
  }

#define ENUM_MACRO(name, v1, v2, v3)               \
  enum class name                                  \
  {                                                \
    v1,                                            \
    v2,                                            \
    v3,                                            \
  };                                               \
  const char *name##Strings[] = {#v1, #v2, #v3};   \
  template <typename T>                            \
  constexpr const char *name##ToString(T value)    \
  {                                                \
    return name##Strings[static_cast<int>(value)]; \
  }

SMO_MACRO(Smoother,
          GLOBAL,
          FUSED_L,
          ConflictFree,
          TensorCore,
          ExactRes,
          Chebyshev);
LA_MACRO(Laplace, Basic, BasicCell, ConflictFree, TensorCore, TensorCoreMMA);
ENUM_MACRO(DoFLayout, DGQ, Q, RT);
ENUM_MACRO(Granularity, none, user_define, multiple);

namespace Util
{
  std::string
  get_filename()
  {
    std::ostringstream oss;

    std::string value_type = "";
    if (std::is_same_v<float, CT::VCYCLE_NUMBER_>)
      value_type = "mixed";
    else if (std::is_same_v<double, CT::VCYCLE_NUMBER_>)
      value_type = "double";
    else
      AssertThrow(false, ExcMessage("Invalid Vcycle number type."));

    std::string str_laplace_variant      = "";
    std::string str_smooth_vmult_variant = "";
    std::string str_smooth_inv_variant   = "";

    for (unsigned int k = 0; k < CT::LAPLACE_TYPE_.size(); ++k)
      {
        str_laplace_variant += LaplaceToString(CT::LAPLACE_TYPE_[k]);
        str_laplace_variant += "_";
      }
    for (unsigned int k = 0; k < CT::SMOOTH_VMULT_.size(); ++k)
      {
        str_smooth_vmult_variant += LaplaceToString(CT::SMOOTH_VMULT_[k]);
        str_smooth_vmult_variant += "_";
      }
    for (unsigned int k = 0; k < CT::SMOOTH_INV_.size(); ++k)
      {
        str_smooth_inv_variant += SmootherToString(CT::SMOOTH_INV_[k]);
        str_smooth_inv_variant += "_";
      }
    const auto str_dof_layout  = DoFLayoutToString(CT::DOF_LAYOUT_);
    const auto str_granularity = GranularityToString(CT::GRANULARITY_);

    const auto n_mpi_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    const auto n_coarse    = CT::N_REPLICATE_;

    oss << "poisson";
    oss << std::scientific << std::setprecision(2);
    oss << "_" << CT::DIMENSION_ << "D";
    oss << "_" << str_dof_layout;
    oss << CT::FE_DEGREE_ << "_";
    oss << n_mpi_procs << "prcs_";
    oss << n_coarse << "cell";
    oss << "_" << str_laplace_variant;
    oss << str_smooth_vmult_variant;
    oss << str_smooth_inv_variant;
    oss << str_granularity;
    oss << "_" << value_type;
    oss << "_" << CT::SETS_;

    return oss.str();
  }

  std::string
  generic_info_to_fstring()
  {
    std::string value_type = "";
    if (std::is_same_v<float, CT::VCYCLE_NUMBER_>)
      value_type = "float";
    else if (std::is_same_v<double, CT::VCYCLE_NUMBER_>)
      value_type = "double";
    else
      AssertThrow(false, ExcMessage("Invalid Vcycle number type."));

    int            devID;
    cudaDeviceProp deviceProp;
    // get number of SMs on this GPU
    AssertCuda(cudaGetDevice(&devID));
    AssertCuda(cudaGetDeviceProperties(&deviceProp, devID));

    auto mps   = deviceProp.multiProcessorCount;
    auto cores = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);

    const auto n_mpi_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    const auto n_coarse    = CT::N_REPLICATE_;

    std::ostringstream oss;

    oss << "Date: " << Utilities::System::get_date() << std::endl
        << "Time: " << Utilities::System::get_time() << std::endl
        << "Git - GPUTPS version: " << GIT_COMMIT_HASH << std::endl
        << "Git - GPUTPS branch: " << GIT_BRANCH << std::endl
        << std::endl;

    oss << "> Device " << devID << ": " << deviceProp.name << std::endl
        << "> SM Capability " << deviceProp.major << "." << deviceProp.minor
        << " detected:" << std::endl
        << "> " << deviceProp.name << " has " << mps << " MP(s) x " << cores
        << " (Cores/MP) = " << mps * cores << " (Cores)" << std::endl
        << std::endl;

    oss << "Settings of parameters: " << std::endl
        << "Dimension:                      " << CT::DIMENSION_ << std::endl
        << "Polynomial degree:              " << CT::FE_DEGREE_ << std::endl
        << "MPI rank(s):                    " << n_mpi_procs << std::endl
        << "Number of coarse cell(s):       " << n_coarse << std::endl
        << "DoF Layout:                     "
        << DoFLayoutToString(CT::DOF_LAYOUT_) << std::endl
        << "Number type for V-cycle:        " << value_type << std::endl;
    oss << "Laplace Variant:                ";
    for (unsigned int k = 0; k < CT::LAPLACE_TYPE_.size(); ++k)
      oss << LaplaceToString(CT::LAPLACE_TYPE_[k]) << " ";
    oss << std::endl;
    oss << "Smoother Vmult Variant:         ";
    for (unsigned int k = 0; k < CT::SMOOTH_VMULT_.size(); ++k)
      oss << LaplaceToString(CT::SMOOTH_VMULT_[k]) << " ";
    oss << std::endl;
    oss << "Smoother Inverse Variant:       ";
    for (unsigned int k = 0; k < CT::SMOOTH_INV_.size(); ++k)
      oss << SmootherToString(CT::SMOOTH_INV_[k]) << " ";
    oss << std::endl;
    oss << "Granularity Scheme:             "
        << GranularityToString(CT::GRANULARITY_) << std::endl
        << "Experiment set:                 " << CT::SETS_ << std::endl
        << "Maximum size:                   " << CT::MAX_SIZES_ << std::endl
        << std::endl;


    return oss.str();
  }

} // namespace Util


#endif // APP_UTILITIES_H
