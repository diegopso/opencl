#ifndef MY_SEMBLANCE_H__
#define MY_SEMBLANCE_H__

typedef struct my_aperture my_aperture_t;
typedef struct my_su_trace my_su_trace_t;

#define TRACES_MAX_SIZE 116
#define DATA_MAX_SIZE 2

struct my_su_trace {
  float data[DATA_MAX_SIZE];
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
  my_su_trace_t traces[TRACES_MAX_SIZE];
};

float my_semblance_2d (my_aperture_t *ap, float A, float B, float C, float D, float E,
	      float t0, float m0, float h0, float *stack);


#endif /* MY_SEMBLANCE_H__ */
