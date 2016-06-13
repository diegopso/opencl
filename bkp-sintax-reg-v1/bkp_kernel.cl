#include <my_semblance.h>

static float my_get_scalco(my_su_trace_t tr) {
	if (tr.scalco == 0)
		return 1;
	if (tr.scalco > 0)
		return tr.scalco;
	return 1.0f / tr.scalco;
}

void my_su_get_midpoint(my_su_trace_t tr, float *mx, float *my) {
	float s = my_get_scalco(tr);
	*mx = s * (tr.gx + tr.sx) * 0.5;
	*my = s * (tr.gy + tr.sy) * 0.5;
}

void my_su_get_halfoffset(my_su_trace_t tr, float *hx, float *hy) {
	float s = my_get_scalco(tr);
	*hx = s * (tr.gx - tr.sx) * 0.5;
	*hy = s * (tr.gy - tr.sy) * 0.5;
}

float sqrtf(float x);

/* The moveout time function tells the time when a wave, propagating from
 * (m0,h0) at t0 to the tace */
static float time_2d(float A, float B, float C, float D, float E, float t0,
		float m0, float m, float h0, float h) {
	float dm = m - m0;
	float dh = h - h0;

	float t2 = t0 + (A * dm) + (B * dh);
	t2 = t2 * t2 + C * dh * dh + D * dm * dm + E * dh * dm;

	if (t2 < 0)
		return -1;
	else
		return sqrtf(t2);
}

float interpol_linear(float x0, float x1, float y0, float y1, float x) {
	return (y1 - y0) * (x - x0) / (x1 - x0) + y0;
}

/*
 * This method computes how much the given parameters fit a collection of traces
 * from the aperture. The 'stack' is the average of the values from the traces
 * intersected by the fitted curve
 */
float my_semblance_2d(my_aperture_t *ap, float A, float B, float C, float D,
		float E, float t0, float m0, float h0, float *stack) {

	/* Get the sample rate from the first trace inside the aperture,
	 it is the same value for all other traces */
	my_su_trace_t tr = ap->traces[0];
	float dt = (float) tr.dt / 1000000;
	float idt = 1 / dt;

	int traces_len = ap->traces_len;

	/* Calculate coherence window (number of trace samples in the trace to
	 include in the semblance) */
	int tau = (((int )(ap->ap_t * idt))>(0)?((int )(ap->ap_t * idt)):(0));
	int w = 2 * tau + 1;

	/* Calculate the semblance  */

	float num[50];
	float den[50];
	for (int i = 0; i < w; i++) {
		num[i] = 0;
		den[i] = 0;
	}

	int M = 0, skip = 0;
	float _stack = 0;

	int i = 0;

	for (i = 0; i < traces_len; i++) {
		tr = ap->traces[i];

		/* Get the trace coordinates in the midpoint and halfoffset spaces */
		float mx, my, hx, hy;
		my_su_get_midpoint(tr, &mx, &my);
		my_su_get_halfoffset(tr, &hx, &hy);

		/* Compute the moveout time ignoring mx and hx because the data is 2D */
		float t = time_2d(A, B, C, D, E, t0, m0, my, h0, hy);
		int it = (int) (t * idt);

		/* Check if the time belongs to the range of the trace */
		if (it - tau >= 0 && it + tau < tr.ns) {
			for (int j = 0; j < w; j++) {
				int k = it + j - tau;
				float v = interpol_linear(k, k + 1, tr.data[k], tr.data[k + 1],
						t * idt + j - tau);
				num[j] += v;
				den[j] += v * v;
				_stack += v;
			}
			M++;
		} else if (++skip == 2) {
			/* Allow only one trace to be excluded from the semblance
			 computation, otherwise the precision of the metric will
			 be compromised */
			goto error;
		}
	}

	float sem = 0;
	float aux = 0;
	for (int j = 0; j < w; j++) {
		sem += num[j] * num[j];
		aux += den[j];
	}

	if (stack) {
		_stack /= M * w;
		*stack = _stack;
	}

	if (aux == 0)
		return 0;

	return sem / aux / M;

	error: return 0;
}

__kernel void foo(
		my_aperture_t *ap, const float m0, const float h0, const float t0,
		__global float *p0, __global float *p1, __global float *np, __global float *out,
		__local float *_Aopt, __local float *_Bopt, __local float *_Copt, __local float *_Dopt, __local float *_Eopt,
		__local float *_stack, __local float *smax)
{
	int ia = get_global_id(0);
	int ib = get_global_id(1);
	int ic = get_global_id(2);

	float a = p0[0] + ((float)ia / (float)np[0]) * (p1[0]-p0[0]);
	float b = p0[1] + ((float)ib / (float)np[1])*(p1[1]-p0[1]);
	float c = p0[2] + ((float)ic / (float)np[2])*(p1[2]-p0[2]);

	for (int id = 0; id < np[3]; id++)
	{

		float d = p0[3] + ((float)id / (float)np[3])*(p1[3]-p0[3]);
		for (int ie = 0; ie < np[4]; ie++)
		{

			float e = p0[4] + ((float)ie / (float)np[4])*(p1[4]-p0[4]);
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

	barrier(CLK_LOCAL_MEM_FENCE);

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
