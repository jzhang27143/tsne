EXECUTABLE := tsne
LDFLAGS=-L/usr/local/depot/cuda-10.2/lib64/
CU_FILES   := tsne.cu
CU_DEPS    :=
CC_FILES   := #tsne.cpp

all: $(EXECUTABLE)

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -g -std=c++11
HOSTNAME=$(shell hostname)

LIBS       :=
FRAMEWORKS :=

NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -std=c++11 -rdc=true
LIBS += GL glut cudart cudadevrt

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

OBJS=$(OBJDIR)/tsne.o $(OBJDIR)/gradients.o $(OBJDIR)/perplexity_search.o $(OBJDIR)/utils.o $(OBJDIR)/quad_tree.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS) *.ppm

#$(EXECUTABLE): dirs $(OBJS)
#		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)
$(EXECUTABLE): dirs $(OBJS)
		$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) $(LDLIBS) -c -o $@

$(OBJDIR)/%.o: kernels/%.cu
		$(NVCC) $< $(NVCCFLAGS) $(LDLIBS) -c -o $@
