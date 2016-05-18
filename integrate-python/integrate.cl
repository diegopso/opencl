__kernel void integrate(__global float *a, unsigned int n)
{
  	int i = get_global_id(0), range;
	float x = 1 / n * (i + 0.5);
	a[i] = 4.0/(1.0 + x*x);

	barrier();

	range = ceil(n/10);
	while(range > 10) {
		reduce(i, range, n, a);
		barrier();
		n = range;
		range = ceil(n/10);
	}
}

void reduce(const int i, const int range, const unsigned int n, __global float *a)
{
	if(i < range) {
		int j, limit = (i + 1) * range;
		float tmp = 0.0f;

		for(j = i * range; j < limit && j < n; j++) {
			tmp += a[j];			
		}

		a[i] = tmp;
	}
}
