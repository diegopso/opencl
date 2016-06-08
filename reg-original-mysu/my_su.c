#include "my_su.h"

static float get_scalco(my_su_trace_t *tr)
{
	if (tr->scalco == 0)
		return 1;
	if (tr->scalco > 0)
		return tr->scalco;
	return 1.0f / tr->scalco;
}

void my_su_get_midpoint(my_su_trace_t *tr, float *mx, float *my)
{
	float s = get_scalco(tr);
	*mx = s * (tr->gx + tr->sx) * 0.5;
	*my = s * (tr->gy + tr->sy) * 0.5;
}

void my_su_get_halfoffset(my_su_trace_t *tr, float *hx, float *hy)
{
	float s = get_scalco(tr);
	*hx = s * (tr->gx - tr->sx) * 0.5;
	*hy = s * (tr->gy - tr->sy) * 0.5;
}