import numpy as np

def read_benchmark_data(target, files, op_type = "Ax"):
    n = 6
    start = 2
    if op_type == "smooth":
        n = 7
        start = 3
    
    data = []

    for filename in files:
        # open the file and read its content
        with open(filename) as f:
            content = f.readlines()

        # extract the relevant data from the content
        local_data = []
        for line in content:
            for name in target:
                if name in line:
                    if len(line.split()) == n:
                        time, dof, s_dof, mem = line.split()[start:]
                        local_data.append([float(time), float(dof), float(s_dof), int(mem)])
        data.append(local_data)
    # convert the data to a numpy array
    data = np.array(data)

    return data