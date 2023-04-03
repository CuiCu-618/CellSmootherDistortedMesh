import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from postprocess import read_benchmark_data
import itertools

files = []
for i in range(1, 8):
    local = []
    for proc in range(1,5):
        local.append(f'../build_DG/BENCHMARK_PARALLEL/Benchmark_poisson_3D_DGQ{i}_{proc}prcs_1cell_ConflictFree_ConflictFree_ConflictFree_multiple_double.log')
    files.append(local)

Data_DP = []

target_smooth = {f'ConflictFree ConflictFree DP'}
for i in range(0,7):
    local = read_benchmark_data(target_smooth, files[i], "smooth")
    Data_DP.append(local)

Data_DP = np.array(Data_DP)

Data_SP = []

target_smooth = {f'ConflictFree ConflictFree SP'}
for i in range(0,7):
    local = read_benchmark_data(target_smooth, files[i], "smooth")
    Data_SP.append(local)

Data_SP = np.array(Data_SP)

procs = np.array([1,2,3,4])


plt.subplot(121)
plt.subplots_adjust(wspace = 0.3)
fig = plt.gcf()
ax = plt.gca()
fig.set_figheight(5)
fig.set_figwidth(12)
fig.set_dpi(300)

marker = itertools.cycle(('1', '+', '.', 'o', '*','d','^')) 
for i in range(0,7):
    plt.plot(procs, Data_DP[i,:,:,2], marker=next(marker), label=f'p={i+1}', mfc='none')

csfont = {'fontname':'Times New Roman', 'size': 18}
plt.title('Double Precision',**csfont)
# plt.yscale('log')
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

marker = itertools.cycle(('1', '+', '.', 'o', '*','d','^')) 
for i in range(0,7):
    plt.plot(procs, Data_SP[i,:,:,2], marker=next(marker), label=f'p={i+1}', mfc='none')

csfont = {'fontname':'Times New Roman', 'size': 18}
plt.title('Single Precision',**csfont)
# plt.yscale('log')
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
           fancybox=True, frameon=True, shadow=False, ncol=4, prop={'family': 'Times New Roman', 'size': 16})
legend.get_frame().set_edgecolor('k') 

plt.subplots_adjust(bottom=0.3)

fig.savefig('../build_DG/Figures/smooth_kernels_mpi.png', dpi=fig.dpi)
fig.savefig('../build_DG/Figures/smooth_kernels_mpi.pdf', dpi=fig.dpi)

# plt.show()
