#ifndef MY_SU_H__
#define MY_SU_H__

typedef struct my_su_trace my_su_trace_t;

#define DATA_MAX_SIZE 2048

struct my_su_trace {
	int sx;
	int sy;
	int gx;
	int gy;
	unsigned short ns;
	unsigned short dt;
	short scalco;
	float data[DATA_MAX_SIZE];
};

/*
 * Returns the midpoint of `tr'.
 */
void my_su_get_midpoint(my_su_trace_t *tr, float *mx, float *my);

/*
 * Returns the half offset of `tr'.
 */
void my_su_get_halfoffset(my_su_trace_t *tr, float *hx, float *hy);


#endif
