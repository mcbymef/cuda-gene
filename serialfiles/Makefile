all: serial matrix

serial:	serialdist.c
	gcc serialdist.c -lm -g -std=c99 -O3 -o serial-dist 

matrix: serialdist-matrix.c
	gcc serialdist-matrix.c -lm -g -std=c99 -O3 -o matrixdist

gperfmatrix:	serialdist-matrix.c
	gcc serialdist-matrix.c -lm -g -std=c99 -O3 -o matrixdist_gperf -lprofiler

clean:
	rm serial-dist matrixdist
