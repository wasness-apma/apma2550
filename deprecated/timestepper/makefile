# Compiler
CXX = g++

# Compiler flags
_INCLUDES = opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/
INCLUDES = $(patsubst %, -I /%, $(_INCLUDES))
CXXFLAGS = -Wall -g $(INCLUDES)

# # Target executable
TARGET = ex/example.out

# # For deleting the target
TARGET_DEL = ex/example.out

# Source files
SDIR = src
SRCS = example.cpp convolution.cpp quadrature.cpp heun.cpp

# Object files
ODIR = obj
OBJS = $(patsubst %, $(ODIR)/%, $(SRCS:.cpp=.o))


# # Default rule to build and run the executable
all: $(TARGET) run

# # Rule to link object files into the target executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Rule to compile .cpp files into .o files
$(ODIR)/%.o: $(SDIR)/%.cpp 
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to run the executable
run: $(TARGET)
	$(TARGET)

# Clean rule to remove generated files
clean:
	rm -f $(ODIR)/*.o $(TARGET_DEL)
	rm -f $(ODIR)/*.o