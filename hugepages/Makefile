# Binary file
BIN = tiny_manna #tiny_manna_con tiny_manna_sin

# Flags
CXXFLAGS = -Wall -Wextra -Werror -std=c++0x -Ofast -flto -march=native # -mavx2
#CXXFLAGS = -std=c++0x -fast #icc compiling options

# Compilers
CXX = g++-8

.PHONY: all clean

all: $(BIN)

# Rules
#tiny_manna_con: tiny_manna_con.cpp huge-alloc.cpp
#       $(CXX) $(CXXFLAGS) -o $@ $^

tiny_manna: tiny_manna.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ -DN=$(N)

clean:
	rm -f $(BIN) *.o *.dat
