#!/bin/bash                                                                                                                           

# Delete makefiles first and then all cmakefiles regarding the previous
# initializiation. This guarantees a fresh build to keep track of current
# GIT-versions.
make clean
find $PWD -iwholename '*cmake*' -not -name CMakeLists.txt -delete

# cmake setup                                                                                                                         
cmake -DCMAKE_BUILD_TYPE="Release" \
      -DDEAL_II_DIR=/export/home/cucui/SimServ06/lib/dealii_cudampi_11_8/ \
      ..