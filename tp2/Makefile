# Binary file
BIN = tiny_manna

# Flags
CXXFLAGS = -Wall -Wextra -Werror -std=c++0x -Ofast -flto -march=native -fopt-info-vec #-DDEBUG # -mavx2
#CXXFLAGS = -std=c++0x -fast #icc compiling options

# Compilers
CXX = g++-8

# Default N
N?=1048576

# Seed. 0 = random
SEED?=0

.PHONY: all clean

all: $(BIN)

# Rules
tiny_manna: tiny_manna.cpp #huge-alloc.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ -DN=$(N) -DSEED=$(SEED) #-DHUGE

clean:
	rm -f $(BIN) *.o *.dat
