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

def read_covergence_data(files):

    data = []

    for filename in files:
        # open the file and read its content
        with open(filename) as f:
            content = f.readlines()

        # extract the relevant data from the content
        local_data = []

        for line in content:
            line = line.strip()
            if line and line[0].isdigit():
                values = [float(x) for x in line.split()]
                local_data.append(values)
        data.append(np.array(local_data))
    # convert the data to a numpy array
    data = np.array(data, dtype=object)

    return data

def write_convergence_table(filename, dataset, type = "it"):

    n_p = dataset.shape[0]
    n_row = int(dataset[0].shape[0] / 2)

    if type == "it":
        col = 11
    elif type == "frac":
        col = 12
    
    file1 = filename + ".txt"

    # Open the file in write mode
    with open(file1, "w") as f:
        # Global kernel
        # Write the LaTeX table header
        f.write("\\begin{table}[tp]\n")
        f.write("\\centering\n")
        f.write("\\renewcommand{\\arraystretch}{1.5}\n")
        f.write("\\begin{tabular}{c|" + "c"*n_p + "}\n")
        f.write("\\hline\n")

        f.write("$L$ &" + " & ".join(["$\mathbb{Q}_" + str(x) + "$" for x in range(1,n_p+1)]) + " \\\\\n")
        f.write("\\hline\n")

        # Write the table data
        for i in range(n_row):
            row_val = []
            row_val.append(int(dataset[0][i,0]))
            for k in range(n_p):
                if i < dataset[k].shape[0] / 2:
                    val = int(dataset[k][i,col]) if type == "it" else round(dataset[k][i,col],1)
                else :
                    val = "---"
                row_val.append(val)
            f.write(" & ".join([str(x) for x in row_val]) + " \\\\\n")

        # Write the LaTeX table footer
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{My table}\n")
        f.write("\\label{tab:my_table}\n")
        f.write("\\end{table}\n")

        f.write("\n")

        # CF kernel
        # Write the LaTeX table header
        f.write("\\begin{table}[tp]\n")
        f.write("\\centering\n")
        f.write("\\renewcommand{\\arraystretch}{1.5}\n")
        f.write("\\begin{tabular}{c|" + "c"*n_p + "}\n")
        f.write("\\hline\n")
    
        f.write("$L$ &" + " & ".join(["$\mathbb{Q}_" + str(x) + "$" for x in range(1,n_p+1)]) + " \\\\\n")
        f.write("\\hline\n")
        
        # Write the table data
        for i in range(n_row):
            row_val = []
            row_val.append(int(dataset[0][i,0]))
            for k in range(n_p):
                shift = int(dataset[k].shape[0] / 2)
                if i < shift:
                    val = int(dataset[k][i+shift,col]) if type == "it" else round(dataset[k][i+shift,col],1)
                else :
                    val = "---"
                row_val.append(val)
            f.write(" & ".join([str(x) for x in row_val]) + " \\\\\n")

        # Write the LaTeX table footer
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{My table}\n")
        f.write("\\label{tab:my_table}\n")
        f.write("\\end{table}\n")