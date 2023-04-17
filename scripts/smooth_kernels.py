import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from postprocess import read_benchmark_data
import os

def barplot(x, y1, y2, y3):
    plt.bar(x-0.3, y1, color='y', alpha=0.5, edgecolor='y', width=0.30, label='GLOBAL')
    plt.bar(x, y2, color='red', alpha=0.5, edgecolor='r', width=0.30, label='FUSED')
    plt.bar(x+0.3, y3, color='blue', alpha=0.5, edgecolor='blue', width=0.30, label='ConflictFree')


def plotfig(dim):
    directory = '../build_DG/BENCHMARK_SERIAL/'
    files = []

    for i in range(1, 11):
        for filename in os.listdir(directory):
            if filename.startswith(f'Benchmark_poisson_{dim}D_DGQ{i}'):
                files.append(os.path.join(directory, filename))
                break

    precision = "DP"
    target_smooth = {f'Basic GLOBAL {precision}', f'Basic FUSED_L {precision}', f'ConflictFree ConflictFree {precision}'}
    smooth_DP = read_benchmark_data(target_smooth, files, "smooth")

    precision = "SP"
    target_smooth = {f'Basic GLOBAL {precision}', f'Basic FUSED_L {precision}', f'ConflictFree ConflictFree {precision}'}
    smooth_SP = read_benchmark_data(target_smooth, files, "smooth")

    # extract the array
    Basic_ = smooth_DP[:,0,:]
    BasicCell_ = smooth_DP[:,1,:]
    CF_ = smooth_DP[:,2,:]

    Basic_perf = Basic_[:,1]
    BasicCell_perf = BasicCell_[:,1]
    CF_perf = CF_[:,1]

    fe = np.arange(1, len(files) + 1)   


    csfont = {'fontname':'Times New Roman', 'size': 18}
    plt.subplot(121)
    plt.subplots_adjust(wspace = 0.3)

    fig = plt.gcf()
    ax = plt.gca()
    fig.set_figheight(5)
    fig.set_figwidth(12)
    fig.set_dpi(300)    

    barplot(fe, Basic_perf, BasicCell_perf, CF_perf)

    plt.title('Double Precision',**csfont)
    # plt.yscale('log')
    plt.grid(linestyle='dashed',axis='y')
    plt.xlabel('Polynomial degree',**csfont)
    plt.ylabel('DoF / s',**csfont)
    plt.xticks(fe)
    plt.tick_params(labelsize=14)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 
    ax.yaxis.set_major_formatter(formatter) 


    # extract the array
    Basic_ = smooth_SP[:,0,:]
    BasicCell_ = smooth_SP[:,1,:]
    CF_ = smooth_SP[:,2,:]

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
    plt.xticks(fe)
    plt.tick_params(labelsize=14)

    barplot(fe, Basic_perf, BasicCell_perf, CF_perf)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 
    ax.yaxis.set_major_formatter(formatter) 

    legend = plt.legend(loc='upper center', bbox_to_anchor=(-0.15, -0.2),
            fancybox=True, frameon=True, shadow=False, ncol=3, prop={'family': 'Times New Roman', 'size': 16})
    legend.get_frame().set_edgecolor('k') 

    plt.subplots_adjust(bottom=0.3)

    fig.savefig(f'../build_DG/Figures/smooth_kernels_{dim}D.png', dpi=fig.dpi)
    fig.savefig(f'../build_DG/Figures/smooth_kernels_{dim}D.pdf', dpi=fig.dpi)

    # plt.show()    

# plotfig(2)
plotfig(3)
