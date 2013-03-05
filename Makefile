all: genedistance 

genedistance: genedist.cu genedist.h
	nvcc -use_fast_math -g -arch=sm_21 genedist.cu -O2 -o GeneDistance -lprofiler

clean:
	rm GeneDistance 
