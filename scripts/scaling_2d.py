import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from postprocess import read_covergence_data
import itertools

files_3d = []
for i in range(1, 11):
    files = []
    files.append(f'../build_DG/TESTING_DOUBLE_SERIAL/poisson_2D_DGQ{i}_1prcs_1cell_ConflictFree_ConflictFree_GLOBAL_ConflictFree_multiple_double.log')
    for p in range(2, 5):
        files.append(f'../build_DG/TESTING_PARALLEL/poisson_2D_DGQ{i}_{p}prcs_1cell_ConflictFree_ConflictFree_ConflictFree_multiple_double.log')
    files_3d.append(files)

data_3d = []
for i in range(1, 11):
    data = read_covergence_data(files_3d[i-1])
    data_3d.append(data)

perf_3d_s = []
for i in range(0, 10):
    perf = []
    for p in range(0,4):
        perf.append(data_3d[i][p][-1,-2])
    perf_3d_s.append(perf)


files_3d = []
for i in range(1, 11):
    files = []
    files.append(f'../build_DG/TESTING_DOUBLE_SERIAL/poisson_2D_DGQ{i}_1prcs_1cell_ConflictFree_ConflictFree_GLOBAL_ConflictFree_multiple_double.log')
    for p in range(2, 5):
        files.append(f'../build_DG/TESTING_PARALLEL/poisson_2D_DGQ{i}_{p}prcs_{p}cell_ConflictFree_ConflictFree_ConflictFree_multiple_double.log')
    files_3d.append(files)


data_3d = []
for i in range(1, 11):
    data = read_covergence_data(files_3d[i-1])
    data_3d.append(data)

perf_3d_w = []
for i in range(0, 10):
    perf = []
    for p in range(0,4):
        perf.append(data_3d[i][p][-1,-2])
    perf_3d_w.append(perf)

procs = np.array([1,2,3,4])


plt.subplot(121)
plt.subplots_adjust(wspace = 0.3)
fig = plt.gcf()
ax = plt.gca()
fig.set_figheight(5)
fig.set_figwidth(12)
fig.set_dpi(300)

marker = itertools.cycle(('1', '+', '.', 'o', '*','d','^','s','x','P')) 
for i in range(0,10):
    plt.plot(procs, perf_3d_s[i], marker=next(marker), label=f'p={i+1}', mfc='none')

csfont = {'fontname':'Times New Roman', 'size': 18}
plt.title('Strong scaling',**csfont)
plt.yscale('log')
plt.grid(linestyle='dashed',axis='y')
plt.xlabel('GPUs',**csfont)
plt.ylabel('s / DoF',**csfont)
plt.xticks(procs)
plt.tick_params(labelsize=14)

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
ax.yaxis.set_major_formatter(formatter) 


plt.subplot(122)
ax = plt.gca()

marker = itertools.cycle(('1', '+', '.', 'o', '*','d','^','s','x','P')) 
for i in range(0,10):
    plt.plot(procs, perf_3d_w[i], marker=next(marker), label=f'p={i+1}', mfc='none')

csfont = {'fontname':'Times New Roman', 'size': 18}
plt.title('Weak scaling',**csfont)
plt.yscale('log')
plt.grid(linestyle='dashed',axis='y')
plt.xlabel('GPUs',**csfont)
plt.ylabel('s / DoF',**csfont)
plt.xticks(procs)
plt.tick_params(labelsize=14)

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
ax.yaxis.set_major_formatter(formatter)


legend = plt.legend(loc='upper center', bbox_to_anchor=(-0.15, -0.2),
           fancybox=True, frameon=True, shadow=False, ncol=5, prop={'family': 'Times New Roman', 'size': 16})
legend.get_frame().set_edgecolor('k') 

plt.subplots_adjust(bottom=0.3)

fig.savefig('../build_DG/Figures/scaling_2d.png', dpi=fig.dpi)
fig.savefig('../build_DG/Figures/scaling_2d.pdf', dpi=fig.dpi)