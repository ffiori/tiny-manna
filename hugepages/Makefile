# Binary file
BIN = tiny_manna_con tiny_manna_sin

# Flags
CXXFLAGS = -Wall -Wextra -Werror -std=c++0x -Ofast -flto -march=native # -mavx2
#CXXFLAGS = -std=c++0x -fast #icc compiling options

# Compilers
CXX = g++

# Default N
N?=1048576

# Seed. 0 = random
SEED?=0

.PHONY: all clean

all: $(BIN)

# Rules
tiny_manna_con: tiny_manna.cpp huge-alloc.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ -DN=$(N) -DSEED=$(SEED) -DHUGE

tiny_manna_sin: tiny_manna.cpp huge-alloc.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ -DN=$(N) -DSEED=$(SEED)

clean:
	rm -f $(BIN) *.o *.dat
