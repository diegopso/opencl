float reduce(const int i, int range, const unsigned int n, __global float *a)
{
	if(i < range) {
		range = n / range;

		int j;
		int limit = (i + 1) * range;
		float tmp = 0.0f;

		//printf("%d, %d\n", limit, n);

		for(j = i * range; j < limit && j < n; j++) {
			tmp += a[j];
		}

		return tmp;
	}

	return 0.0f;
}

__kernel void integrate(__global float *a, unsigned int n)
{
  	int i = get_global_id(0);
	float range, tmp, x = 1.0f / ((float) n) * (i + 0.5f);

	a[i] = 4.0f / (1.0f + x*x);
}
