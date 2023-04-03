import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from postprocess import read_benchmark_data

def barplot(x, y1, y2, y3):
    plt.bar(x-0.3, y1, color='y', alpha=0.5, edgecolor='y', width=0.30, label='Basic')
    plt.bar(x, y2, color='red', alpha=0.5, edgecolor='r', width=0.30, label='Basic Cell')
    plt.bar(x+0.3, y3, color='blue', alpha=0.5, edgecolor='blue', width=0.30, label='ConflictFree')

files2d = []
for i in range(1, 11):
    files2d.append(f'../build_DG/BENCHMARK_SERIAL/Benchmark_poisson_2D_DGQ{i}_1prcs_1cell_Basic_BasicCell_ConflictFree_Basic_ConflictFree_GLOBAL_FUSED_L_ConflictFree_multiple_double.log')

files3d = []
for i in range(1, 8):
    files3d.append(f'../build_DG/BENCHMARK_SERIAL/Benchmark_poisson_3D_DGQ{i}_1prcs_1cell_Basic_BasicCell_ConflictFree_Basic_ConflictFree_GLOBAL_FUSED_L_ConflictFree_multiple_double.log')

precision = "DP"
target_Ax = {f'Basic {precision}', f'BasicCell {precision}', f'ConflictFree {precision}'}
Ax_DP_3d = read_benchmark_data(target_Ax, files3d)

precision = "SP"
target_Ax = {f'Basic {precision}', f'BasicCell {precision}', f'ConflictFree {precision}'}
Ax_SP_3d = read_benchmark_data(target_Ax, files3d)

# extract the array
Basic_ = Ax_DP_3d[:,0,:]
BasicCell_ = Ax_DP_3d[:,1,:]
CF_ = Ax_DP_3d[:,2,:]

Basic_perf = Basic_[:,1]
BasicCell_perf = BasicCell_[:,1]
CF_perf = CF_[:,1]


fe_2d = np.array([1,2,3,4,5,6,7,8,9,10])
fe_3d = np.array([1,2,3,4,5,6,7])



csfont = {'fontname':'Times New Roman', 'size': 18}
plt.subplot(121)
plt.subplots_adjust(wspace = 0.3)

fig = plt.gcf()
ax = plt.gca()
fig.set_figheight(5)
fig.set_figwidth(12)
fig.set_dpi(300)

barplot(fe_3d, Basic_perf, BasicCell_perf, CF_perf)

plt.title('Double Precision',**csfont)
# plt.yscale('log')
plt.grid(linestyle='dashed',axis='y')
plt.xlabel('Polynomial degree',**csfont)
plt.ylabel('DoF / s',**csfont)
plt.xticks(fe_3d)
plt.tick_params(labelsize=14)

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
ax.yaxis.set_major_formatter(formatter) 


# extract the array
Basic_ = Ax_SP_3d[:,0,:]
BasicCell_ = Ax_SP_3d[:,1,:]
CF_ = Ax_SP_3d[:,2,:]

Basic_perf = Basic_[:,1]
BasicCell_perf = BasicCell_[:,1]
CF_perf = CF_[:,1]

plt.subplot(122)
ax = plt.gca()

plt.title('Single Precision',**csfont)
# plt.yscale('log')
plt.grid(linestyle='dashed',axis='y')
plt.xlabel('Polynomial degree',**csfont)
plt.ylabel('DoF / s',**csfont)
plt.xticks(fe_3d)
plt.tick_params(labelsize=14)

barplot(fe_3d, Basic_perf, BasicCell_perf, CF_perf)

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
ax.yaxis.set_major_formatter(formatter) 

legend = plt.legend(loc='upper center', bbox_to_anchor=(-0.15, -0.2),
           fancybox=True, frameon=True, shadow=False, ncol=3, prop={'family': 'Times New Roman', 'size': 16})
legend.get_frame().set_edgecolor('k') 

plt.subplots_adjust(bottom=0.3)

fig.savefig('../build_DG/Figures/laplace_kernels.png', dpi=fig.dpi)
fig.savefig('../build_DG/Figures/laplace_kernels.pdf', dpi=fig.dpi)

# plt.show()
