#!/bin/bash                                                                                                                           

cd ..
make release
cd -

# DOF=16777216
# DOF=134217728
# for DOF in 4096 32768 262144 2097152 16777216 134217728

# variant="Basic ConflictFree"
variant="TensorCore TensorCoreMMA TensorCoreMMACF"

for DOF in 50000000
do
for p in 3 
do
for v in $variant
do
    echo "/////////////////////"
    echo "Starting 3D degree=$p Dof=$DOF $v"
    echo "/////////////////////"
    python3 ../scripts/ct_parameter.py -DIM 3 -DEG $p -MAXSIZE $DOF -REDUCE 1e-8 -MAXIT 10 \
          -LA $v -VNUM double -SETS none -G none 
    cd ..
    make biharm 
    cd -
    echo "/////////////////////"
    echo "Running 3D degree=$p Dof=$DOF $v"
    echo "/////////////////////"
    # ../apps/biharm -device=2
    ncu -k regex:laplace_kernel -s 3 -c 2 \
      --metrics gpc__cycles_elapsed.avg.per_second,\
gpu__time_duration.sum,\
dram__bytes.sum,\
dram__bytes.sum.per_second,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
sm__inst_executed_pipe_fp64.sum,\
sm__inst_executed_pipe_fma.sum,\
sm__inst_executed_pipe_fp16.sum,\
sm__inst_executed_pipe_tensor_op_dmma.sum,\
sm__inst_executed_pipe_tensor_op_hmma.sum,\
sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
lts__t_sectors_lookup_hit.sum,\
lts__t_sectors_lookup_miss.sum,\
l1tex__t_sectors_lookup_hit.sum,\
l1tex__t_sectors_lookup_miss.sum,\
launch__registers_per_thread,\
launch__grid_size,\
launch__block_size,\
launch__shared_mem_per_block_dynamic,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__sass_inst_executed_op_local_ld.sum,\
smsp__sass_inst_executed_op_local_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.pct_of_peak_sustained_elapsed,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum.pct_of_peak_sustained_elapsed,\
l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed \
        ../apps/biharm -device=2 > ${v}.log 
    # ncu -f -o ${v} -k regex:laplace_kernel -s 4 -c 2 --set full \
    #     --import-source no ../apps/biharm -device=2
      
done
done
done

# for DOF in 50000000
# do
# for d in "${dis[@]}"
# do
# for p in 3 7 # 3 4 5 6 7
# do
#     echo "/////////////////////"
#     echo "Starting 3D degree=$p Dof=$DOF Dis=$d"
#     echo "/////////////////////"
#     python3 ../scripts/ct_parameter.py -DIM 3 -DEG $p -MAXSIZE $DOF -REDUCE 1e-8 -MAXIT 200 \
#           -DIS $d -FACE element_wise_partial -LA Basic -SMV Basic -SMI MCS \
#           -REP 1 -VNUM double -SETS error_analysis -G none
#     cd ..
#     make poisson 
#     cd -
#     echo "/////////////////////"
#     echo "Running 3D degree=$p Dof=$DOF"
#     echo "/////////////////////"
#     # ../apps/poisson -device=2
#     ncu -k regex:apply_kernel_shmem \
#       --metrics gpc__cycles_elapsed.avg.per_second,\
# gpu__time_duration.sum,\
# dram__bytes.sum,\
# dram__bytes.sum.per_second,\
# dram__bytes_read.sum,\
# dram__bytes_write.sum,\
# sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed,\
# sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
# sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
# sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
# sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
# l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
# l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
# lts__t_sectors_lookup_hit.sum,\
# lts__t_sectors_lookup_miss.sum,\
# l1tex__t_sectors_lookup_hit.sum,\
# l1tex__t_sectors_lookup_miss.sum,\
# launch__registers_per_thread,\
# sm__warps_active.avg.pct_of_peak_sustained_active,\
# smsp__sass_inst_executed_op_local_ld.sum,\
# smsp__sass_inst_executed_op_local_st.sum,\
# l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.pct_of_peak_sustained_elapsed,\
# l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum.pct_of_peak_sustained_elapsed,\
# l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed \
#         ../apps/poisson -device=2 > Ax_element_partial_Q${p}  
# done
# done
# done
#
