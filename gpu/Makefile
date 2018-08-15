CXX=nvcc
CXXFLAGS=-std=c++11 -arch=sm_35 --ptxas-options=-v --compiler-options "-Ofast -Wall" #-arch=sm_52 #for titan x
#~ CXXFLAGS=-std=c++11 -arch=sm_35 --ptxas-options=-v -G -lineinfo # para debuguear con cuda-memcheck
LDFLAGS=

TARGETS=tiny_manna

all: $(TARGETS)

tiny_manna: tiny_manna.cu
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: clean all

clean:
	rm -f *.o $(TARGETS)
