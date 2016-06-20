#include <my_semblance.h>

#ifndef MAX
# define MAX(a, b) ((a)>(b)?(a):(b))
#endif

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

	/* Get the sample rate from the first trace inside the aperture,
	 it is the same value for all other traces */
	__global my_su_trace_t *tr = &ap->traces[0];
	float dt = (float) tr->dt / 1000000;
	float idt = 1 / dt;

	/* Calculate coherence window (number of trace samples in the trace to
	 include in the semblance) */
	int tau = MAX((int)(ap->ap_t * idt), 0);
	int w = 2 * tau + 1;

	/* Calculate the semblance  */

	float num[10];
	float den[10];
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
		float my, hy;
		float s;

		if (tr->scalco == 0)
			s = 1;
		else if (tr->scalco > 0)
			s = tr->scalco;
		else
			s = 1.0f / tr->scalco;

		s = s * 0.5;
		my = s * (tr->gy + tr->sy);
		hy = s * (tr->gy - tr->sy);

		float t;

		/* Compute the moveout time ignoring mx and hx because the data is 2D */
		float dm = my - m0;
		float dh = hy - h0;

		float t2 = t0 + (A * dm) + (B * dh);
		t2 = t2 * t2 + C * dh * dh + dm * (D * dm + E * dh);

		if (t2 < 0)
			t = -1;
		else
			t = sqrt(t2);

		int it = (int)(t * idt);
		float temp = t*idt - it;
		int k = it - tau - 1;

		/* Check if the time belongs to the range of the trace */
		if (it - tau >= 0 && it + tau < tr->ns) {
			for (int j = 0; j < w; j++) {
				k++;
				float v = (tr->data[k+1] - tr->data[k]) * temp + tr->data[k];

				num[j] += v;
				den[j] += v*v;
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
		_stack /= M*w;
		*stack = _stack;
	}

	if (aux == 0)
		return 0;

	return sem / aux / M;

	error:
	return 0;
}
