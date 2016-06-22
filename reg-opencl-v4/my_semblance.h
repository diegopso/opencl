#ifndef MY_SEMBLANCE_H__
#define MY_SEMBLANCE_H__

typedef struct my_aperture my_aperture_t;
typedef struct my_su_trace my_su_trace_t;

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


#endif /* MY_SEMBLANCE_H__ */
