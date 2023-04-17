import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from postprocess import read_covergence_data, write_convergence_table

files_2d_d = []
files_2d_f = []

for i in range(1, 11):
    files_2d_d.append(f'../build_DG/TESTING_DOUBLE_SERIAL/poisson_2D_DGQ{i}_1prcs_1cell_ConflictFree_ConflictFree_GLOBAL_ConflictFree_multiple_double.log')
    files_2d_f.append(f'../build_DG/TESTING_FLOAT_SERIAL/poisson_2D_DGQ{i}_1prcs_1cell_ConflictFree_ConflictFree_GLOBAL_ConflictFree_multiple_mixed.log')

data_2d_d = read_covergence_data(files_2d_d)
data_2d_f = read_covergence_data(files_2d_f)

write_convergence_table("../build_DG/TESTING_DOUBLE_SERIAL/table_2d", data_2d_d, "frac")

files_3d_d = []
files_3d_f = []

for i in range(1, 8):
    files_3d_d.append(f'../build_DG/TESTING_DOUBLE_SERIAL/poisson_3D_DGQ{i}_1prcs_1cell_ConflictFree_ConflictFree_GLOBAL_ConflictFree_multiple_double.log')
    files_3d_f.append(f'../build_DG/TESTING_FLOAT_SERIAL/poisson_3D_DGQ{i}_1prcs_1cell_ConflictFree_ConflictFree_GLOBAL_ConflictFree_multiple_mixed.log')

data_3d_d = read_covergence_data(files_3d_d)
data_3d_f = read_covergence_data(files_3d_f)

write_convergence_table("../build_DG/TESTING_DOUBLE_SERIAL/table_3d", data_3d_d, "frac")

def barplot(x, y1, y2, y3):
    plt.bar(x-0.3, y1, color='y', alpha=0.5, edgecolor='y', width=0.30, label='GLOBAL')
    plt.bar(x, y2, color='red', alpha=0.5, edgecolor='r', width=0.30, label='ConflictFree DP')
    plt.bar(x+0.3, y3, color='blue', alpha=0.5, edgecolor='blue', width=0.30, label='ConflictFree SP')

perf_2d_g = []
perf_2d_d = []
perf_2d_f = []

for i in range(0,10):
    n_row = int(data_2d_d[i].shape[0]/2)
    perf_2d_g.append(data_2d_d[i][n_row-1,-2])
    perf_2d_d.append(data_2d_d[i][-1,-2])
    perf_2d_f.append(data_2d_f[i][-1,-2])

perf_3d_g = []
perf_3d_d = []
perf_3d_f = []

for i in range(0,7):
    n_row = int(data_3d_d[i].shape[0]/2)
    perf_3d_g.append(data_3d_d[i][n_row-1,-2])
    perf_3d_d.append(data_3d_d[i][-1,-2])
    perf_3d_f.append(data_3d_f[i][-1,-2])

plt.subplot(121)
plt.subplots_adjust(wspace = 0.3)
fig = plt.gcf()
ax = plt.gca()
fig.set_figheight(5)
fig.set_figwidth(12)
fig.set_dpi(300)

fe = np.arange(1,11)

barplot(fe,perf_2d_g,perf_2d_d,perf_2d_f)

csfont = {'fontname':'Times New Roman', 'size': 18}
plt.title('2D',**csfont)
# plt.yscale('log')
plt.grid(linestyle='dashed',axis='y')
plt.xlabel('Polynomial degree',**csfont)
plt.ylabel('s / DoF',**csfont)
plt.xticks(fe)
plt.tick_params(labelsize=14)

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
ax.yaxis.set_major_formatter(formatter) 


plt.subplot(122)
ax = plt.gca()

fe = np.arange(1,8)

barplot(fe,perf_3d_g,perf_3d_d,perf_3d_f)

csfont = {'fontname':'Times New Roman', 'size': 18}
plt.title('3D',**csfont)
# plt.yscale('log')
plt.grid(linestyle='dashed',axis='y')
plt.xlabel('Polynomial degree',**csfont)
plt.ylabel('s / DoF',**csfont)
plt.xticks(fe)
plt.tick_params(labelsize=14)

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
ax.yaxis.set_major_formatter(formatter)


legend = plt.legend(loc='upper center', bbox_to_anchor=(-0.15, -0.2),
           fancybox=True, frameon=True, shadow=False, ncol=4, prop={'family': 'Times New Roman', 'size': 16})
legend.get_frame().set_edgecolor('k') 

plt.subplots_adjust(bottom=0.3)

fig.savefig('../build_DG/Figures/convergence.png', dpi=fig.dpi)
fig.savefig('../build_DG/Figures/convergence.pdf', dpi=fig.dpi)
