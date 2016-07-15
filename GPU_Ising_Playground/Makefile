CUDA_PATH = /usr/local/cuda-6.5
CC = g++ 
NVCC = $(CUDA_PATH)/bin/nvcc
CFLAGS = -L$(CUDA_PATH)/lib64 -Wl,-rpath -Wl,$(CUDA_PATH)/lib64 -lcudart -std=c++0x
NVCCFLAGS= -arch=compute_20 -code=sm_20 -I$(CUDA_SDK_PATH)/C/common/inc --ptxas-options=-v -fmad=true -use_fast_math
COPTFLAGS = -O3
LDFLAGS =

CUSRCS = heisenberg.cu
CSRCS = heisenberg.cpp

CUOBJS = $(CUSRCS:.cu=.o__cu)
COBJS = $(CSRCS:.c=.o__c)

magnet: $(CUOBJS) $(COBJS)
	$(CC) $(CFLAGS) $^ -o $@ 

%.o__c: %.c
	$(CC) -o $@ -c $<

%.o__cu: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

clean:
	rm -f core *.o__cu *.o__c *~ magnet
