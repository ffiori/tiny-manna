#!/usr/bin/env python3

slots = 0
singleperf = {}

print("slots,threads,var,value")

threads = "\"SIMD single thread\""
with open("../simd/results-simd.txt", "r") as f:
    for line in f:
        if "slots" in line:
            slots = int(line.split(" ")[10])
        elif "instructions" in line:
            ipc = float(line.split("#")[1].strip().split(" ")[0])
            print("{0},{1},ipc,{2}".format(slots, threads, ipc))
        elif "cache-misses" in line:
            miss = float(line.split("#")[1].strip().split("%")[0])
            print("{0},{1},cachemiss,{2}".format(slots, threads, miss))
        elif "elapsed" in line:
            time = float(line.strip().split(" ")[0])
            singleperf[slots] = slots/time
            print("{0},{1},normtime,{2}".format(slots, threads, slots/time))
            efficiency = (slots/time) / singleperf[slots]
            print("{0},{1},efficiency,{2}".format(slots, threads, efficiency))

threads = 0
threadstr = ""

with open("results-openmp.txt", "r") as f:
    for line in f:
        if "slots" in line:
            slots = int(line.split(" ")[10])
        elif "Using" in line:
            threads = int(line.split(" ")[2])
            threadstr = str(threads)
            if threads<10:
                threadstr = "0"+threadstr
        elif "instructions" in line:
            ipc = float(line.split("#")[1].strip().split(" ")[0])
            print("{0},{1},ipc,{2}".format(slots, threadstr, ipc))
        elif "cache-misses" in line:
            miss = float(line.split("#")[1].strip().split("%")[0])
            print("{0},{1},cachemiss,{2}".format(slots, threadstr, miss))
        elif "elapsed" in line:
            time = float(line.strip().split(" ")[0])
            print("{0},{1},normtime,{2}".format(slots, threadstr, slots/time))

            efficiency = (slots/time) / singleperf[slots] / threads
            print("{0},{1},efficiency,{2}".format(slots, threadstr, efficiency))

# ./parse.py > melted.csv
# Rscript plot-openmp.r
