#include <my_semblance.h>

#ifndef MAX
# define MAX(a, b) ((a)>(b)?(a):(b))
#endif

static float my_get_scalco(__global my_su_trace_t *tr) {
	if (tr->scalco == 0)
		return 1;
	if (tr->scalco > 0)
		return tr->scalco;
	return 1.0f / tr->scalco;
}

void my_su_get_midpoint(__global my_su_trace_t *tr, float *mx, float *my) {
	float s = my_get_scalco(tr);
	*mx = s * (tr->gx + tr->sx) * 0.5;
	*my = s * (tr->gy + tr->sy) * 0.5;
}

void my_su_get_halfoffset(__global my_su_trace_t *tr, float *hx, float *hy) {
	float s = my_get_scalco(tr);
	*hx = s * (tr->gx - tr->sx) * 0.5;
	*hy = s * (tr->gy - tr->sy) * 0.5;
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
float my_semblance_2d(__global my_aperture_t *ap,
		float A, float B, float C, float D, float E,
		float t0, float m0, float h0,
		float *stack)
{
    printf("inicio my_semblance\n");

	/* Get the sample rate from the first trace inside the aperture,
	 it is the same value for all other traces */
	__global my_su_trace_t *tr = &ap->traces[0];
	printf("my_tr: ", tr->dt);
	float dt = (float) tr->dt / 1000000;
	float idt = 1 / dt;

	/* Calculate coherence window (number of trace samples in the trace to
	 include in the semblance) */
	int tau = MAX((int)(ap->ap_t * idt), 0);
	int w = 2 * tau + 1;

	/* Calculate the semblance  */

	float num[50];
	float den[50];
	for(int i=0;i<w;i++) {
		num[i]=0;
		den[i]=0;
	}

	int M = 0, skip = 0;
	float _stack = 0;

	int len = ap->traces_len;
	for (int i = 0; i < len; i++) {
		tr = &ap->traces[i];

		/* Get the trace coordinates in the midpoint and halfoffset spaces */
		float mx, my, hx, hy;
		my_su_get_midpoint(tr, &mx, &my);
		my_su_get_halfoffset(tr, &hx, &hy);

		/* Compute the moveout time ignoring mx and hx because the data is 2D */
		float t = time_2d(A, B, C, D, E, t0, m0, my, h0, hy);
		int it = (int)(t * idt);

		/* Check if the time belongs to the range of the trace */
		if (it - tau >= 0 && it + tau < tr->ns) {
			for (int j = 0; j < w; j++) {
				int k = it + j - tau;
				float v = interpol_linear(k, k+1,
						tr->data[k], tr->data[k+1],
						t*idt + j - tau);
				num[j] += v;
				den[j] += v*v;
				_stack += v;
			}
			M++;
		} else if (++skip == 2) {
			/* Allow only one trace to be excluded from the semblance
			 computation, otherwise the precision of the metric will
			 be compromised */
			printf("error\n");
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
		_stack /= M*w;
		*stack = _stack;
	}

	if (aux == 0)
	return 0;

	return sem / aux / M;

	error:
	return 0;
}
