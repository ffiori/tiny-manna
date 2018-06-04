# Binary file
BIN = tiny_manna

# Flags
CFLAGS = -Wall -Wextra -Werror -std=c++0x
LDFLAGS =

# Compilers
CPP = g++
LINKER = g++

# Files
MAKEFILE = Makefile
CPP_SOURCES = $(BIN).cpp
HEADERS =
CPP_OBJS = $(patsubst %.cpp, %.o, $(CPP_SOURCES))

# Rules
$(BIN): $(CPP_OBJS) $(HEADERS) $(MAKEFILE)
	$(LINKER) -o $(BIN) $(CPP_OBJS) $(LDFLAGS) $(INCLUDES) $(LIBS)

$(CPP_OBJS): $(CPP_SOURCES) $(HEADERS) $(MAKEFILE)
	$(CPP) -c $(CPP_SOURCES) $(CFLAGS) $(INCLUDES) $(PARAMETERS)

clean:
	rm -f $(BIN) *.o *.dat
