#ifndef MY_SEMBLANCE_H__
#define MY_SEMBLANCE_H__

typedef struct my_aperture my_aperture_t;
typedef struct my_su_trace my_su_trace_t;

struct my_aperture
{
  float ap_t;
  my_su_trace_t* traces;
};

struct my_su_trace {
  float *data;
  unsigned short ns;
  unsigned short dt;
  int sx;
  int sy;
  int gx;
  int gy;
  short scalco;
};

float semblance_2d (my_aperture_t *ap, float A, float B, float C, float D, float E,
	      float t0, float m0, float h0, float *stack);


#endif /* MY_SEMBLANCE_H__ */
