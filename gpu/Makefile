CXX=nvcc
CXXFLAGS=-std=c++11 --ptxas-options=-v --compiler-options "-Ofast -Wall"
CXXFLAGS+= -gencode=arch=compute_35,code=sm_35 # K40
CXXFLAGS+= -gencode=arch=compute_52,code=sm_52 # Titan X
CXXFLAGS+= -rdc=true -lcudadevrt
#~ CXXFLAGS+= -G -lineinfo # para debuguear con cuda-memcheck
LDFLAGS=

TARGETS=tiny_manna

all: $(TARGETS)

%: %.cu
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: clean all

clean:
	rm -f *.o $(TARGETS)
