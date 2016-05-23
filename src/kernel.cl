#include <semblance.h>

__kernel void foo(
    aperture_t *ap, const float m0, const float h0, const float t0,
    __global float *p0,  __global float *p1, __global float *np, __global float *out, 
    __local float *_Aopt, __local float *_Bopt, __local float *_Copt, __local float *_Dopt, __local float *_Eopt,
    __local float *_stack, __local float *smax
) {
    int ia = get_global_id(0);    
    int ib = get_global_id(1);
    int ic = get_global_id(2);

    float a = p0[0] + ((float)ia / (float)np[0]) * (p1[0]-p0[0]);
    float b = p0[1] + ((float)ib / (float)np[1])*(p1[1]-p0[1]);
    float c = p0[2] + ((float)ic / (float)np[2])*(p1[2]-p0[2]);

	for (int id = 0; id < np[3]; id++) {

	    float d = p0[3] + ((float)id / (float)np[3])*(p1[3]-p0[3]);
	    for (int ie = 0; ie < np[4]; ie++) {

		    float e = p0[4] + ((float)ie / (float)np[4])*(p1[4]-p0[4]);
		    float st;
		    /* Check the fit of the parameters to the data and update the 
		     * maximum for that point if necessary */
		    float s = semblance_2d(ap, a, b, c, d, e, t0, m0, h0, &st);
		    if (s > smax[ia]) {
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
    for (int ia = 0; ia < np[0]; ia++) {
        if (smax[ia] > ssmax) {
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
