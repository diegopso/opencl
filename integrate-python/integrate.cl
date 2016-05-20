int myCeil(float k) {
    int l = (int)k;    
    if(k == (float)l)
        return l;
    return l + 1;
}

float reduce(const int i, int group, const unsigned int n, __global float *a)
{
	if(i < group) {
		int range = myCeil(((float) n) / group);

		int j;
		int limit = (i + 1) * range;
		float tmp = 0.0f;

		for(j = i * range; j < limit && j < n; j++) {
			tmp += a[j];
		}

		return tmp;
	}

	return 0.0f;
}

__kernel void integrate(__global float *a, unsigned int n)
{
  	int i = get_global_id(0), group;
	float tmp, x = 1.0f / ((float) n) * (i + 0.5f);

	a[i] = 4.0f / (1.0f + x*x);
	
	barrier(CLK_GLOBAL_MEM_FENCE);

	group = myCeil(0.1f * n);
	while(group >= 10.0f) {
		tmp = reduce(i, group, n, a);
		
		barrier(CLK_GLOBAL_MEM_FENCE);
		
		a[i] = tmp;
		
		barrier(CLK_GLOBAL_MEM_FENCE);
		
		n = group;
		group = myCeil(0.1f * n);
	}
}
