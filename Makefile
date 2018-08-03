# Binary file
BIN = tiny_manna

# Flags
CXXFLAGS = -Wall -Wextra -Werror -std=c++0x -Ofast -flto -march=native # -mavx2
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
tiny_manna: tiny_manna.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ -DN=$(N) -DSEED=$(SEED)

clean:
	rm -f $(BIN) *.o *.dat
