CXX=nvcc
CXXFLAGS=-std=c++11 -arch=sm_35 --ptxas-options=-v --compiler-options "-Ofast -Wall -flto" #-arch=sm_52 #for titan x
LDFLAGS=

TARGETS=tiny_manna

all: $(TARGETS)

tiny_manna: tiny_manna.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: clean all

clean:
	rm -f *.o $(TARGETS)
