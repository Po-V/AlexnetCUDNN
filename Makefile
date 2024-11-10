NVCC = nvcc
CXXFLAGS = -std=c++11 -O3 -arch=sm_75 -g
LDFLAGS = -lcudnn -lcudart -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui
INCLUDES = -I/usr/local/cuda/include \
		   -I/usr/local/include/opencv4 \
		   -I/usr/include/opencv4 \
		   -I/usr/local/cuda-12.6/include \
		   -I/usr/include \
		   -I/usr/local/include \
		   -I/usr/local/cuda/lib64

LIBS = -lcublas -lcurand -lcudnn

# Target and source
TARGET = alexnet
SRC = alexnet.cu

# Default make target
all: $(TARGET)

# Build the executable
$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SRC) $(LIBS) $(LDFLAGS)

# Run compute-sanitizer
sanitizer: $(TARGET)
	compute-sanitizer --tool memcheck ./$(TARGET)

# Clean up generated files
clean:
	rm -f $(TARGET)
