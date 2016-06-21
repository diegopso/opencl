#ifndef MY_SEMBLANCE_H__
#define MY_SEMBLANCE_H__

typedef struct my_aperture my_aperture_t;
typedef struct my_su_trace my_su_trace_t;

#define TRACES_MAX_SIZE 116
#define DATA_MAX_SIZE 2502

struct my_su_trace {
  float data[DATA_MAX_SIZE];
  unsigned short ns;
  unsigned short dt;
  int sy;
  int gy;
  short scalco;
};

struct my_aperture
{
  float ap_t;
  my_su_trace_t traces[TRACES_MAX_SIZE];
  int traces_len;
};


float my_semblance_2d (__global my_aperture_t *ap, float A, float B, float C, float D, float E,
	      float t0, float m0, float h0, float *stack);


#endif /* MY_SEMBLANCE_H__ */
