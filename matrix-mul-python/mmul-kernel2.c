__kernel void mmul(const unsigned int N, __global float *A,__global float  *B,  __global float *C) {
	int	j, k;
	int	i = get_global_id(0);
	float tmp;

	if (i < N) {
		for(j = 0; j < N; j++) {
			tmp = 0.0f;
			for(k=0;k<N;k++) {
				tmp	+= A[i*N+k]*B[k*N+j];
			}
			C[i*N+j]+=tmp;
			//printf("i[%d]j[%d]: tmp: %f ----- C:%f \n",i,j, tmp, C[i*N+j]);
		}
	}
}
