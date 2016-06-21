#include <my_semblance_kernel.h>
#include <my_semblance.c>

__kernel void calculate(
		__global my_aperture_t *ap,
		__global float *p0,
		__global float *p1,
		__global int *np,

		__global float *_Aopt,
		__global float *_Bopt,
		__global float *_Copt,
		__global float *_Dopt,
		__global float *_Eopt,
		__global float *_stack,
		__global float *out,
		const float m0,
		const float h0,
		const float t0,
		__global float *smax)
{

	int ia = get_global_id(0);
	int ib = get_global_id(1);
	int ic = get_global_id(2);

	float a = p0[0] + ((float)ia / (float)np[0]) * (p1[0]-p0[0]);
	float b = p0[1] + ((float)ib / (float)np[1]) * (p1[1]-p0[1]);
	float c = p0[2] + ((float)ic / (float)np[2]) * (p1[2]-p0[2]);

	float p03 = p0[3];
	float mul_id = (p1[3] - p03 ) / (float)np[3];
	float p04 = p0[4];
	float mul_ie = (p1[4]-p0[4]) / (float)np[4];

	int np3 = np[3];
	int np4 = np[4];

	float l_Aopt = 0;
	float l_Bopt = 0;
	float l_Copt = 0;
	float l_Dopt = 0;
	float l_Eopt = 0;
	float l_stack = 0;
	float l_smax = smax[ia];

	float d = p03;
	float e;

	for (int id = 0; id < np3; id++)
	{
		e = p04;

		for (int ie = 0; ie < np4; ie++)
		{

			float st;
			float s = my_semblance_2d(ap, a, b, c, d, e, t0, m0, h0, &st);

			if (s > l_smax)
			{
				l_Aopt = a;
				l_Bopt = b;
				l_Copt = c;
				l_Dopt = d;
				l_Eopt = e;
				l_stack = st;
				l_smax = s;
			}
			e += mul_ie;
		}
		d += mul_id;
	}

	if(l_smax > smax[ia]) {
		_Aopt[ia] = l_Aopt;
		_Bopt[ia] = l_Bopt;
		_Copt[ia] = l_Copt;
		_Dopt[ia] = l_Dopt;
		_Eopt[ia] = l_Eopt;
		_stack[ia] = l_stack;
		smax[ia] = l_smax;
	}

	/* Now find the best fit between different 'A' values */
	float ssmax = -1.0;
	for (int ia = 0; ia < np[0]; ia++)
	{
		if (smax[ia] > ssmax)
		{
			out[0] = _Aopt[ia];
			out[1] = _Bopt[ia];
			out[2] = _Copt[ia];
			out[3] = _Dopt[ia];
			out[4] = _Eopt[ia];
			out[5] = _stack[ia];
			out[6] = smax[ia];
			ssmax = smax[ia];
		}
	}
}

