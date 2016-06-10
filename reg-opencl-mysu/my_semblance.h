#ifndef MY_SEMBLANCE_H__
#define MY_SEMBLANCE_H__

#include <my_su.h>

#define TRACES_MAX_SIZE 116

typedef struct my_aperture my_aperture_t;

struct my_aperture {
    float ap_m, ap_h, ap_t;
    my_su_trace_t traces[TRACES_MAX_SIZE];
};

float my_semblance_2d(my_aperture_t *ap,
        float A, float B, float C, float D, float E,
        float t0, float m0, float h0, float *stack);

#endif

