#include <my_semblance.h>

#ifndef MAX
# define MAX(a, b) ((a)>(b)?(a):(b))
#endif

float my_get_scalco(my_su_trace_t tr) {
	if (tr.scalco == 0)
		return 1;
	if (tr.scalco > 0)
		return tr.scalco;
	return 1.0f / tr.scalco;
}

void my_su_get_midpoint(my_su_trace_t tr, float mx, float my) {
	float s = my_get_scalco(tr);

}

void my_su_get_halfoffset(my_su_trace_t tr, float hx, float hy) {
	float s = my_get_scalco(tr);
}

float sqrtf(float x);

/* The moveout time function tells the time when a wave, propagating from
 * (m0,h0) at t0 to the tace */
float time_2d(float A, float B, float C, float D, float E, float t0,
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

float my_semblance_2d(my_aperture_t *ap, float A, float B, float C, float D,
		float E, float t0, float m0, float h0, float *stack) {

	my_su_trace_t tr = ap->traces[0];
	float dt = (float) tr.dt / 1000000;
	float idt = 1 / dt;

	int tau = MAX((int )(ap->ap_t * idt), 0);
	int w = 2 * tau + 1;

	float num[50];
	float den[50];
	for (int i = 0; i < w; i++) {
		num[i] = 0;
		den[i] = 0;
	}
	int M = 0, skip = 0;
	float _stack = 0;

	int len = sizeof(ap->traces) / sizeof(ap->traces[0]);

	int i = 0;


	float mx, my, hx, hy, t, v;

	int it, j;

	for (i = 0; i < len; i++) {
		tr = ap->traces[i];

		my_su_get_midpoint(tr, mx, my);
		my_su_get_halfoffset(tr, hx, hy);




		if (it - tau >= 0 && it + tau < tr.ns) {
			for (j = 0; j < w; j++) {
				int k = it + j - tau;
				v = interpol_linear(k, k + 1, tr.data[k], tr.data[k + 1],
						t * idt + j - tau);
				num[j] += v;
				den[j] += v * v;
				_stack += v;
			}
			M = M + 1;
		} else if (++skip == 2) {
			return 0;
		}


	}


	float sem = 0;
	float aux = 0;
	for (j = 0; j < w; j++) {
		sem += num[j] * num[j];
		aux += den[j];
	}

	if (aux == 0) {
		return 0;
	}

	return sem / aux / M;

}

__kernel void foo(__global my_aperture_t *ap, __global float *out) {

	int i=get_global_id(0);
	int j=get_global_id(1);

	out[0] = 0;
	out[1] = 0;
	out[2] = 0;
	out[3] = 0;
	out[4] = 0;
	out[5] = 0;
	out[6] = 0;
}
