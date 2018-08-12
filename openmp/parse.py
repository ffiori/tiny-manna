#!/usr/bin/env python3

slots = 0
threads = 0

singleperf = 0

print("slots,threads,var,value")

with open("results-openmp.txt", "r") as f:
    for line in f:
        if "slots" in line:
            slots = int(line.split(" ")[10])
        elif "Using" in line:
            threads = int(line.split(" ")[2])
        elif "instructions" in line:
            ipc = float(line.split("#")[1].strip().split(" ")[0])
            print("{0},{1},ipc,{2}".format(slots, threads, ipc))
        elif "cache-misses" in line:
            miss = float(line.split("#")[1].strip().split("%")[0])
            print("{0},{1},cachemiss,{2}".format(slots, threads, miss))
        elif "elapsed" in line:
            time = float(line.strip().split(" ")[0])
            print("{0},{1},normtime,{2}".format(slots, threads, slots/time))
            if threads == 1:
                singleperf = slots/time

            efficiency = (slots/time) / singleperf / threads
            print("{0},{1},efficiency,{2}".format(slots, threads, efficiency))

threads = "\"1, AVX2\""
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
            print("{0},{1},normtime,{2}".format(slots, threads, slots/time))
            efficiency = (slots/time) / singleperf
            print("{0},{1},efficiency,{2}".format(slots, threads, efficiency))
