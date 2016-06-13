#include <my_semblance.h>

#define SQR(x) ((x)*(x))
#ifndef MAX
# define MAX(a, b) ((a)>(b)?(a):(b))
#endif

static float get_scalco(my_su_trace_t *tr)
{
	if (tr->scalco == 0)
		return 1;
	if (tr->scalco > 0)
		return tr->scalco;
	return 1.0f / tr->scalco;
}

void my_su_get_midpoint(my_su_trace_t *tr, float *mx, float *my)
{
	float s = get_scalco(tr);
	*mx = s * (tr->gx + tr->sx) * 0.5;
	*my = s * (tr->gy + tr->sy) * 0.5;
}

void my_su_get_halfoffset(my_su_trace_t *tr, float *hx, float *hy)
{
	float s = get_scalco(tr);
	*hx = s * (tr->gx - tr->sx) * 0.5;
	*hy = s * (tr->gy - tr->sy) * 0.5;
}

static float my_time_2d(float A, float B, float C, float D, float E, float t0, float m0, float m, float h0, float h)
{
	float dm, dh, t2;
	dm = m - m0;
	dh = h - h0;

	t2 = t0 + (A*dm) + (B*dh);
	t2 = t2*t2 + C*dh*dh + D*dm*dm + E*dh*dm;

	if (t2 < 0)
		return -1;
	else
		return sqrt(t2);
}

float my_interpol_linear(float x0, float x1, float y0, float y1, float x)
{
	return (y1 - y0) * (x - x0) / (x1 - x0) + y0;
}

float my_semblance_2d_cl(__global my_aperture_t *ap, float A, float B, float C, float D, float E, float t0, float m0, float h0, float *stack)
{
	float dt, idt, t;
	float num[500], den[500];
	float _stack = 0;
	float mx, my, hx, hy, v;
	int tau, w, i, it, j, k;
	int M = 0, skip = 0;
	float sem = 0;
	float aux = 0;

	my_su_trace_t tr = ap->traces[0];

	dt = (float) tr.dt / 1000000;
	idt = 1 / dt;

	tau = MAX((int)(ap->ap_t * idt), 0);
	w = 2 * tau + 1;

	for (i = 0; i < TRACES_MAX_SIZE; i++) {
		tr = ap->traces[i];

		mx = 0;
		my = 0; 
		hx = 0;
		hy = 0;

		my_su_get_midpoint(&tr, &mx, &my);
		my_su_get_halfoffset(&tr, &hx, &hy);

		t = my_time_2d(A, B, C, D, E, t0, m0, my, h0, hy);
		it = (int)(t * idt);

		if (it - tau >= 0 && it + tau < tr.ns) {
			for (j = 0; j < w; j++) {
				k = it + j - tau;
				v = my_interpol_linear(k, k+1, tr.data[k], tr.data[k+1], t * idt + j - tau);
				num[j] += v;
				den[j] += v * v;
				_stack += v;
			}
			M++;
		} else if (++skip == 2) {
			return 0;
		}
	}

	for (int j = 0; j < w; j++) {
		sem += num[j] * num[j];
		aux += den[j];
	}

	if (stack) {
		_stack /= M*w;
		*stack = _stack;
	}

	if (aux == 0)
		return 0;

	return sem / aux / M;
}


__kernel void foo(__global my_aperture_t *ap, const float m0,	const float h0,	const float t0,	__global float *p0, __global float *p1, __global int *np, __global float *out,
	__global float *_Aopt, __global float *_Bopt, __global float *_Copt, __global float *_Dopt, __global float *_Eopt, __global float *_stack, __global float *smax)
{
	int ia, ib, ic, id, ie;
	float a, b, c, d, e, s, ssmax;

	ia = get_global_id(0);
	ib = get_global_id(1);
	ic = get_global_id(2);

	a = p0[0] + ((float)ia / (float)np[0]) * (p1[0]-p0[0]);
	b = p0[1] + ((float)ib / (float)np[1]) * (p1[1]-p0[1]);
	c = p0[2] + ((float)ic / (float)np[2]) * (p1[2]-p0[2]);
	
	for (id = 0; id < np[3]; id++)
	{

		d = p0[3] + ((float)id / (float)np[3])*(p1[3]-p0[3]);
		for (ie = 0; ie < np[4]; ie++)
		{

			e = p0[4] + ((float)ie / (float)np[4])*(p1[4]-p0[4]);

			float st;
			s = my_semblance_2d_cl(ap, a, b, c, d, e, t0, m0, h0, &st);

			// printf("%d, %d, %d, %d, %d => %f\n", ia, ib, ic, id, ie, s);
			
			if (s > smax[ia])
			{
				// out[ia] = s;
				// out[ia + 1] = st;
				// out[ia + 2] = a;
				// out[ia + 3] = b;
				// out[ia + 4] = c;
				// out[ia + 5] = d;
				// out[ia + 6] = e;

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

	barrier(CLK_GLOBAL_MEM_FENCE);

	if(ia == 0 && ib == 0 && ic == 0) {
		ssmax = -1.0;
		for (ia = 0; ia < np[0]; ia++)
		{
			// printf("%d, %f, %f\n", ia, _Bopt[ia], smax[ia]);
			if (smax[ia] > ssmax)
			{
				out[0] = _Aopt[ia];
				out[1] = _Bopt[ia];
				out[2] = _Copt[ia];
				out[3] = _Dopt[ia];
				out[4] = _Eopt[ia];
				out[5] = _stack[ia];
				out[6] = smax[ia];

				// out[0] = out[ia];
				// out[1] = out[ia + 1];
				// out[2] = out[ia + 2];
				// out[3] = out[ia + 3];
				// out[4] = out[ia + 4];
				// out[5] = out[ia + 5];
				// out[6] = out[ia + 6];

				ssmax = smax[ia];
			}
		}

	}
}
