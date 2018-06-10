# Binary file
BIN = tiny_manna_con tiny_manna_sin

# Flags
CXXFLAGS = -Wall -Wextra -Werror -std=c++0x -Ofast -flto # -march=native

# Compilers
CXX = g++

.PHONY: all clean

all: $(BIN)

# Rules
tiny_manna_con: tiny_manna_con.cpp huge-alloc.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

tiny_manna_sin: tiny_manna_sin.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(BIN) *.o *.dat
