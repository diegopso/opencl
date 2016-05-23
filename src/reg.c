#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <utils.h>
#include <vector.h>
#include <semblance.h>
#include <su.h>
#include <errno.h>

#include <CL/opencl.h>
//#include <unistd.h>

#define MAXSOURCE 2048
#define MAX_DEVICE_NAME_SIZE 100
#define LOCALSIZE 32

int main(int argc, char *argv[])
{
    int i;
    
    int n = 5;
    /* A, B, C, D, E */
    float p0[n], p1[n];
    int np[n];

    if (argc != 21) {
        fprintf(stderr, "Usage: %s M0 H0 T0 TAU A0 A1 NA B0 B1 NB "
            "C0 C1 NC D0 D1 ND E0 E1 NE INPUT\n", argv[0]);
        exit(1);
    }

    float m0 = atof(argv[1]);
    float h0 = atof(argv[2]);
    float t0 = atof(argv[3]);
    float tau = strtof(argv[4], NULL);

    /* p0 is where the search starts, p1 is where the search ends and np is the 
     * number of points in between p0 and p1 to do the search */   
    for (i = 0; i < 5; i++) {
        p0[i] = atof(argv[5 + 3*i]);
        p1[i] = atof(argv[5 + 3*i + 1]);
        np[i] = atoi(argv[5 + 3*i + 2]);
    }

    /* Load the traces from the file */

    char *path = argv[20];
    FILE *fp = fopen(path, "r");

    if (!fp) {
        fprintf(stderr, "Failed to open prestack file '%s'!\n", path);
        return 1;
    }

    su_trace_t tr;
    vector_t(su_trace_t) traces;
    vector_init(traces);

    while (su_fgettr(fp, &tr)) {
        vector_push(traces, tr);
    }

    /* Construct the aperture structure from the traces, which is a vector
     * containing pointers to traces */

    aperture_t ap;
    ap.ap_m = 0;
    ap.ap_h = 0;
    ap.ap_t = tau;
    vector_init(ap.traces);
    for (int i = 0; i < traces.len; i++)
        vector_push(ap.traces, &vector_get(traces, i));
        
    /*-------------------------------------------------------------------------*/

    char *kernelSource = (char *) malloc(MAXSOURCE * sizeof(char));
    
    FILE * file = fopen("kernel.cl", "r");
    if(file == NULL) {
        printf("Error: open the kernel file (kernel.cl)\n");
        exit(1);
    }
    
    // Read kernel code
    size_t source_size = fread(kernelSource, 1, MAXSOURCE, file);
    
    //Device input buffers
    cl_mem d_p0, d_p1, d_np, d_aopt, d_bopt, d_copt, d_dopt, d_eopt, d_stack, d_smax;
    //Device output buffer
    cl_mem  d_out;
    
    char deviceName[MAX_DEVICE_NAME_SIZE];
    cl_platform_id cpPlatform;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_platform_id *platforms;
    cl_uint platformCount;
    
    //Tamanho em bytes de cada vetor
    size_t bytes_p0 = sizeof(float) * n;
    size_t bytes_p1 = sizeof(float) * n;
    size_t bytes_np = sizeof(float) * n;
    size_t bytes_opt = sizeof(float) * np[0];
    
    //Numero de workitems em cada local work group (local size)
    size_t localSize[3] = {LOCALSIZE, LOCALSIZE, LOCALSIZE};
    size_t globalSize[3] = {
        ceil((float)np[0] / (float)localSize[0]),
        ceil((float)np[1] / (float)localSize[1]),
        ceil((float)np[2] / (float)localSize[2])
    };
    
    //

    /* Find the best parameter combination */

    float a, b, c, d, e, sem, stack;
    compute_max(&ap, m0, h0, t0, p0, p1, np, &a, &b, &c, &d, &e, &sem, &stack);

    printf("A=%g\n", a);
    printf("B=%g\n", b);
    printf("C=%g\n", c);
    printf("D=%g\n", d);
    printf("E=%g\n", e);
    printf("Stack=%g\n", stack);
    printf("Semblance=%g\n", sem);
    printf("\n");

    return 0;
}
