#!/usr/bin/env python3

slots = 0
singleperf = {}

print("slots,what,var,value")

threadstr = ""
with open("../openmp/results-openmp.txt", "r") as f:
    for line in f:
        if "slots" in line:
            slots = int(line.split(" ")[10])
        elif "Using" in line:
            threads = int(line.split(" ")[2])
            if threads == 6:
                threadstr = "OpenMP (6 hilos)"
            elif threads == 11:
                threadstr = "OpenMP (11 hilos)"
            else:
                threadstr = ""
        elif "elapsed" in line:
            time = float(line.strip().split(" ")[0])
            if threadstr != "":
                print("{0},{1},normtime,{2}".format(slots, threadstr, slots/time))

threadstr = "GPU K40"
with open("results-gpu-k40.txt", "r") as f:
    for line in f:
        if "slots" in line:
            slots = int(line.split(" ")[10])
        elif "elapsed" in line:
            time = float(line.strip().split(" ")[0])
            print("{0},{1},normtime,{2}".format(slots, threadstr, slots/time))

threadstr = "GPU Titan X"
with open("results-gpu-titanx.txt", "r") as f:
    for line in f:
        if "slots" in line:
            slots = int(line.split(" ")[10])
        elif "elapsed" in line:
            time = float(line.strip().split(" ")[0])
            print("{0},{1},normtime,{2}".format(slots, threadstr, slots/time))

# ./parse.py > melted.csv
# Rscript plot-cuda.r
