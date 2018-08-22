CXX=nvcc
CXXFLAGS=-std=c++11 -use_fast_math -restrict -Xptxas -O3,-v --compiler-options "-Ofast -Wall"
CXXFLAGS+= -gencode=arch=compute_35,code=sm_35 # K40
CXXFLAGS+= -gencode=arch=compute_52,code=sm_52 # Titan X
CXXFLAGS+= -rdc=true -lcudadevrt # dynamic parallelism
#~ CXXFLAGS+= -G -lineinfo # para debuguear con cuda-memcheck
LDFLAGS=

# Default N
N?=1048576

TARGETS=tiny_manna_ondevice_reduce

all: $(TARGETS)

%: %.cu
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -DNSLOTS=$(N)

.PHONY: clean all

clean:
	rm -f *.o $(TARGETS)
