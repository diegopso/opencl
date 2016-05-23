__kernel void mmul(const unsigned int N, __global float *A,__global float  *B,  __global float *C) {
	int	k;
	int	i=get_global_id(0);
	int	j=get_global_id(1);
	float tmp=0.0f;

	if (i < N && j < N) {

		for(k=0;k<N;k++) {
			tmp	+= A[i*N+k]*B[k*N+j];
		}
		C[i*N+j]+=tmp;
		//printf("i[%d]j[%d]: tmp: %f ----- C:%f \n",i,j, tmp, C[i*N+j]);
	}
}
