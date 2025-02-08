import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from postprocess import read_data, write_convergence_table

files_2d_d = []

for i in range(1, 9):
    files_2d_d.append(f'/export/home/cucui/CLionProjects/GPUTensorProductSmoothers/build_stokes/TEST_ALL/Stokes_2D_RT{i}_Basic_MatrixStruct_Basic_MatrixStruct_GLOBAL_Direct_SchurTensorProduct_none_double_1s.log')

data_2d_d = read_data(files_2d_d)

write_convergence_table("/export/home/cucui/CLionProjects/GPUTensorProductSmoothers/build_stokes/TEST_ALL/table_2d", data_2d_d, 1, 1, "frac", 4)
