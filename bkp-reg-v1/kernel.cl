typedef struct my_su_trace my_su_trace_t;
typedef struct my_aperture my_aperture_t;

struct my_su_trace {
  float data[2];
  unsigned short ns;
  unsigned short dt;
  int sx;
  int sy;
  int gx;
  int gy;
  short scalco;
};

struct my_aperture
{
  float ap_t;
  my_su_trace_t traces[116];
  int traces_len;
};

__kernel void foo(
		__global my_aperture_t *ap, __global float *out) {

			out[0] = 0;
			out[1] = 0;
			out[2] = 0;
			out[3] = 0;
			out[4] = 0;
			out[5] = 0;
			out[6] = 0;
}
