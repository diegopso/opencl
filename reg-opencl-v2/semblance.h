#ifndef SEMBLANCE_H__
#define SEMBLANCE_H__

#include <vector.h>
#include <su.h>

typedef struct aperture aperture_t;

struct aperture {
    float ap_m, ap_h, ap_t;
    vector_t(su_trace_t*) traces;
};


#endif /* SEMBLANCE_H__ */
