#include <my_semblance_kernel.h>
#include <my_semblance.c>

__kernel void calculate(
		__global my_aperture_t *ap,
		const float m0,
		const float h0,
		const float t0,
		__global float *p0,
		__global float *p1,
		__global int *np,
		__global float *out,

		__global float *_Aopt,
		__global float *_Bopt,
		__global float *_Copt,
		__global float *_Dopt,
		__global float *_Eopt,
		__global float *_stack,
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

	float d = p03 - mul_id;
	float e;

	for (int id = 0; id < np3; id++)
	{
		d += mul_id;
		e = p04 - mul_ie;

		for (int ie = 0; ie < np4; ie++)
		{
			e += mul_ie;

			float st;
			/* Check the fit of the parameters to the data and update the
			 * maximum for that point if necessary */
			float s = my_semblance_2d(ap, a, b, c, d, e, t0, m0, h0, &st);

			if (s > smax[ia])
			{
				smax[ia] = s;
				_stack[ia] = st;
				_Aopt[ia] = a;
				_Bopt[ia] = b;
				_Copt[ia] = c;
				_Dopt[ia] = d;
				_Eopt[ia] = e;
			}
		}
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

