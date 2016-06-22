all: reg.c su.c
	gcc -o build/reg reg.c -I. su.c  -lOpenCL -lm -std=gnu99 -D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE