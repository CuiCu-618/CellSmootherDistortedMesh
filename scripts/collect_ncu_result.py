import sys

import numpy as np


def convert_to_Tbytes(value, unit):
    if unit == "Kbyte":
        return float(value) / 1024 / 1024 / 1024
    elif unit == "Mbyte":
        return float(value) / 1024 / 1024
    elif unit == "Gbyte":
        return float(value) / 1024
    else:
        return float(-1)


def convert_to_seconds(value, unit):
    if unit == "usecond":
        return float(value) / 1000000
    elif unit == "msecond":
        return float(value) / 1000
    else:
        return float(value)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py file_path")
        sys.exit(1)

    file_path = sys.argv[1]

    dram_bytes = []
    gpu_time_duration = []
    bank_conflict = []
    fma = []
    fp64 = []
    dmma = []
    hmma = []
    L2_hit = []
    L2_miss = []
    shared_ld = []
    shared_st = []
    L1_peak = []

    fma_inst = []
    fp64_inst = []
    dmma_inst = []
    tb_size = []
    occupancy = []
    reg = []

    L2_flag = 0
    L1_flag = 0

    with open(file_path, "r") as file:
        for line in file:
            if "dram__bytes.sum" in line:
                parts = line.strip().split()
                value = parts[-1]
                unit = parts[-2]
                converted_value = convert_to_Tbytes(value, unit)
                if converted_value > 0:
                    dram_bytes.append(converted_value)
            elif "gpu__time_duration.sum" in line:
                parts = line.strip().split()
                value = parts[-1]
                unit = parts[-2]
                converted_value = convert_to_seconds(value, unit)
                gpu_time_duration.append(converted_value)
            elif "sm__pipe_fma_cycles_active" in line:
                parts = line.strip().split()
                value = parts[-1]
                fma.append(float(value))
            elif "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st" in line:
                parts = line.strip().split()
                value = parts[-1]
                bank_conflict.append(float(value))
            elif "sm__pipe_fp64_cycles_active" in line:
                parts = line.strip().split()
                value = parts[-1]
                fp64.append(float(value))
            elif "sm__pipe_tensor_op_dmma_cycles_active" in line:
                parts = line.strip().split()
                value = parts[-1]
                dmma.append(float(value))
            elif "sm__pipe_tensor_op_hmma_cycles_active" in line:
                parts = line.strip().split()
                value = parts[-1]
                hmma.append(float(value))
            elif "lts__t_sectors_lookup_hit.sum " in line:
                L2_flag = 1
                parts = line.strip().split()
                value = parts[-1]
                L2_hit.append(float(value))
            elif "lts__t_sectors_lookup_miss.sum " in line:
                parts = line.strip().split()
                value = parts[-1]
                L2_miss.append(float(value))
            elif "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum" in line:
                L1_flag = 1
                parts = line.strip().split()
                value = parts[-1]
                shared_ld.append(float(value))
            elif "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum" in line:
                parts = line.strip().split()
                value = parts[-1]
                shared_st.append(float(value))
            elif "l1tex__data_pipe_lsu_wavefronts.avg" in line:
                parts = line.strip().split()
                value = parts[-1]
                L1_peak.append(float(value))
            elif "sm__inst_executed_pipe_fma.sum" in line:
                parts = line.strip().split()
                value = parts[-1]
                fma_inst.append(float(value))
            elif "sm__inst_executed_pipe_fp64.sum" in line:
                parts = line.strip().split()
                value = parts[-1]
                fp64_inst.append(float(value))
            elif "sm__inst_executed_pipe_tensor_op_dmma.sum" in line:
                parts = line.strip().split()
                value = parts[-1]
                dmma_inst.append(float(value))
            elif "launch__registers_per_thread" in line:
                parts = line.strip().split()
                value = parts[-1]
                reg.append(float(value))
            elif "launch__block_size" in line:
                parts = line.strip().split()
                value = parts[-1]
                tb_size.append(float(value))
            elif "sm__warps_active.avg.pct_of_peak_sustained_active" in line:
                parts = line.strip().split()
                value = parts[-1]
                occupancy.append(float(value))

    print("DRAM Bytes [GB]:", end=" ")
    print(" ".join(f"{dram * 1024:.3e}" for dram in dram_bytes), end=" ")
    print(f" | {sum(dram_bytes) * 1024:.3e}")

    print("\nGPU Time Duration [ms]:", end=" ")
    print(" ".join(f"{gpu_time * 1000:.3e}" for gpu_time in gpu_time_duration), end=" ")
    print(f" | {sum(gpu_time_duration) * 1000:.3e}")

    peak_DRAM = 2.039
    print("\nDRAM peak [%]:", end=" ")
    print(
        " ".join(
            f"{dram_bytes[i] / gpu_time_duration[i] / peak_DRAM * 100:.2f}"
            for i in range(len(dram_bytes))
        )
    )
    print("\nBank Conflicts:", end=" ")
    print(" ".join(f"{bank:.0f}" for bank in bank_conflict))

    if L1_flag == 1:
        print("\nL1 peak [%]:          ", end=" ")
        print(" ".join(f"{l1:5.2f}" for l1 in L1_peak))
        print("Shared peak [%]:      ", end=" ")
        print(
            " ".join(
                f"{shared_ld[i] + shared_st[i]:5.2f}" for i in range(len(shared_ld))
            )
        )

    if L2_flag == 1:
        print("L2 cache hit rate [%]:", end=" ")
        print(
            " ".join(
                f"{L2_hit[i] / (L2_hit[i] + L2_miss[i]) * 100 :5.2f}"
                for i in range(len(L2_hit))
            )
        )

    print("\nfma [%]: ", end=" ")
    print(" ".join(f"{perf:5.2f}" for perf in fma))
    print("fp64 [%]:", end=" ")
    print(" ".join(f"{perf:5.2f}" for perf in fp64))
    print("dmma [%]:", end=" ")
    print(" ".join(f"{perf:5.2f}" for perf in dmma))

    perf = []
    ai = []

    peak_fp64 = 8.81
    peak_fp32 = 17.62
    peak_dmma = 17.62
    peak_hmma = 281.92
    peak_tf32 = 140.96

    if "WMMA" in file_path:
        peak_tmp = peak_tf32
    else:
        peak_tmp = peak_hmma

    for i, val in enumerate(fma):
        perf.append(
            fma[i] / 100 * peak_fp32
            + fp64[i] / 100 * peak_fp64
            + dmma[i] / 100 * peak_dmma
        )
        ai.append(perf[i] * gpu_time_duration[i] / dram_bytes[i])

    # print("\nPerf [TFLOP/s]   :", end=" ")
    # print(" ".join(f"{p:5.3f}" for p in perf))
    #
    # print("AI [FLOP/byte]   :", end=" ")
    # print(" ".join(f"{p:5.3f}" for p in ai))

    count_fma = 2
    count_fp64 = 2
    count_dmma = 2 * 8 * 8 * 8 - 8 * 8

    total_flops = []
    new_perf = []
    new_ai = []
    for i, val in enumerate(fma_inst):

        scale = 32 if tb_size[i] > 32 else tb_size[i]
        total_flops.append(
            fma_inst[i] * count_fma * scale * 0
            + fp64_inst[i] * count_fp64 * scale
            + dmma_inst[i] * count_dmma * scale
        )
        new_perf.append(total_flops[i] / 1e12 / gpu_time_duration[i])

        new_ai.append(new_perf[i] * gpu_time_duration[i] / dram_bytes[i])

    print("\nBlock Size:   ", end=" ")
    print(" ".join(f"{p:4.0f}" for p in tb_size))
    print("Number regs.: ", end=" ")
    print(" ".join(f"{p:4.0f}" for p in reg))
    print("Occupancy [%]:", end=" ")
    print(" ".join(f"{p:5.2f}" for p in occupancy))

    print("\nPerf [TFLOP/s]   :", end=" ")
    print(" ".join(f"{p:5.3f}" for p in new_perf), end=" ")
    print(f" | {sum(total_flops) / sum(gpu_time_duration) / 1e12 :5.3f}")

    print("AI [FLOP/byte]   :", end=" ")
    print(" ".join(f"{p:5.3f}" for p in new_ai), end=" ")
    print(f" | {sum(total_flops) / sum(dram_bytes) / 1e12 :5.3f}")

    Bsh = 17.145
    if L1_flag == 1:
        print("SH AI [FLOP/byte]:", end=" ")
        print(
            " ".join(
                f"{new_perf[i] / (shared_ld[i] + shared_st[i]) / Bsh * 100:5.3f}"
                for i in range(len(new_perf))
            ),
            end=" ",
        )
    print(
        f" | {sum(total_flops) / sum(dram_bytes) / 1e12 / ((sum(shared_ld) + sum(shared_st)) * Bsh / 100) :5.3f}"
    )
    print("\n\n")

    # dof = 16777216
    # print("\nFlop/DoF :")
    # for i in range(len(fma)):
    #     print(fp64[i] * peak_fp64 * 1e12 * gpu_time_duration[i] / dof)
    #     # print(perf[i] * 1e12 * gpu_time_duration[i] / dof)
