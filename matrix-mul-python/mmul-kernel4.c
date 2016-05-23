__kernel void mmul(const unsigned int N, __global float *A,__global float  *B,  __global float *C, __local float *Bwrk) {
	int	j, k;
	int	i = get_global_id(0);

	int iloc = get_local_id(0);

	int nloc = get_local_size(0);

	float tmp;
	/*Setup a work array for A in private memory*/
	float awrk[2048];

	for (k = 0; k < N; k++) {
		awrk[k] = A[i*N+k];
	}

	//	if (i < N) {
	for (j = 0; j < N; j++) {

		for(k = iloc; k < N; k+=nloc) {
			Bwrk[k] = B[k * N + j];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		tmp = 0.0f;
		for (k=0;k<N;k++) {
			tmp	+= awrk[k]*B[k*N+j];
		}
		C[i*N+j] += tmp;
		//printf("i[%d]j[%d]: tmp: %f ----- C:%f \n",i,j, tmp, C[i*N+j]);
	}
	//	}
}
