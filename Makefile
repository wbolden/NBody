PROJECTNAME :=NBody

CC := gcc
CFLAGS := -std=c++11 -Wall $(ARGS)

NVCC := /usr/local/cuda/bin/nvcc
NVCCGENCODEFLAGS := -arch=compute_30 -code=sm_30,compute_30

NVCCFLAGS := $(NVCCGENCODEFLAGS) $(ARGS)
NVCCLINKFLAGS := $(NVCCGENCODEFLAGS)

CUDAPATH := /usr/local/cuda
CUDALIBPATH := -L$(CUDAPATH)/lib64

LIBPATH :=
GLPATH := /usr/include
LIBS := -lGLEW -lGL -lGLU -lglfw3 -lX11 -lXxf86vm -lXrandr -lXi -lcuda -lcudart
INCLUDES := -I$(CUDAPATH)/include -I$(GLPATH)

all: $(PROJECTNAME)

$(PROJECTNAME): Main.o Display.o Physics.o
	$(NVCC) $(NVCCLINKFLAGS) -o $@ $^ $(LIBPATH) $(CUDALIBPATH) $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $(INCLUDES) $< 

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $(INCLUDES) $<

.PHONY clean:
	sudo rm *.o
	sudo rm $(PROJECTNAME)