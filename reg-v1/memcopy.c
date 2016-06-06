#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int
main ()
{

  float *array;
  array[0] = 5.1;
  array[1] = 1.2;

  float destfloat[5];

  int i = 0;
  int size = sizeof(array) / sizeof(array[0]);
  printf ("array size: %d \n", size);

  for (i = 0; i < size; i++)
    {
      float *v = malloc (sizeof(float));
      memcpy (v, &array[i], sizeof(float));
      printf ("%f\n", *v);
      destfloat[i] = *v;
    }

  for (i = 0; i < size; i++)
    {
      printf ("destfloat[%d]: %f\n", i, destfloat[i]);
    }

  return (0);
}
