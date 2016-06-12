#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <utils.h>
#include <semblance.h>
#include <my_semblance.h>
#include <su.h>
#include <my_su.h>
#include <errno.h>

#include <CL/opencl.h>
#include <string.h>
#include <unistd.h>


#define MAXSOURCE 5000
#define MAX_DEVICE_NAME_SIZE 100

void transformTr(su_trace_t *tr, my_su_trace_t *mtr)
{
    (*mtr).dt = (*tr).dt;
    (*mtr).ns = (*tr).ns;
    (*mtr).gx = (*tr).gx;
    (*mtr).sx = (*tr).sx;
    (*mtr).gy = (*tr).gy;
    (*mtr).sy = (*tr).sy;
    (*mtr).scalco = (*tr).scalco;

    for (int i = 0; i < DATA_MAX_SIZE; i++) {
        (*mtr).data[i] = (*tr).data[i];
    }
}

/*
 * compute_max finds the best parameters 'Aopt', 'Bopt', 'Copt', 'Dopt' and 'Eopt' 
 * that fit a curve to the data in 'ap' from a reference point (m0, h0, t0). Also 
 * returning its fit (coherence/semblance) through 'sem' and the average of values 
 * along the curve through 'stack'
 *
 * The lower limit for searching each parameter is specified as a element in the 
 * vector 'n0' and the upper limit in vector 'n1', the number of divisions for 
 * the search space is specified through 'np'
 */
void compute_max(my_aperture_t *ap, float m0, float h0, float t0,
    const float n0[5], const float n1[5], const int np[5], float *Aopt,
    float *Bopt, float *Copt, float *Dopt, float *Eopt, float *sem,
    float *stack)
{
    /* The parallel version of the code will compute the best parameters for 
     * each value of the parameter 'A', so we need to store np[0] different 
     * values of each parameter, stack and semblance */
    float _Aopt[np[0]], _Bopt[np[0]], _Copt[np[0]], 
          _Dopt[np[0]], _Eopt[np[0]];
    float smax[np[0]];
    float _stack[np[0]];

    /* Split the outermost loop between threads. Each thread will
     * compute the best fit for a given parameter 'A' value */
    #pragma omp parallel for schedule(dynamic)
    for (int ia = 0; ia < np[0]; ia++) {
        smax[ia] = -1;
        float a = n0[0] + ((float)ia / (float)np[0])*(n1[0]-n0[0]);
        for (int ib = 0; ib < np[1]; ib++) {
            float b = n0[1] + ((float)ib / (float)np[1])*(n1[1]-n0[1]);
            for (int ic = 0; ic < np[2]; ic++) {
                float c = n0[2] + ((float)ic / (float)np[2])*(n1[2]-n0[2]);
                for (int id = 0; id < np[3]; id++) {
                    float d = n0[3] + ((float)id / (float)np[3])*(n1[3]-n0[3]);
                    for (int ie = 0; ie < np[4]; ie++) {
                        float e = n0[4] + ((float)ie / (float)np[4])*(n1[4]-n0[4]);
                        float st;
                        /* Check the fit of the parameters to the data and update the 
                         * maximum for that point if necessary */
                        float s = my_semblance_2d(ap, a, b, c, d, e, t0, m0, h0, &st);
                        if (s > smax[ia]) {
                            smax[ia] = s;
                            _stack[ia] = st;
                            _Aopt[ia] = a;
                            _Bopt[ia] = b;
                            _Copt[ia] = c;
                            _Dopt[ia] = d;
                            _Eopt[ia] = e;
                        }
                    }
                }
            }
            /* Uncomment this to roughly check the progress */
            /* fprintf(stderr, "."); */
        }
    }

    /* Now find the best fit between different 'A' values */
    float ssmax = -1.0;
    *stack = 0;
    for (int ia = 0; ia < np[0]; ia++) {
        if (smax[ia] > ssmax) {
            *Aopt = _Aopt[ia];
            *Bopt = _Bopt[ia];
            *Copt = _Copt[ia];
            *Dopt = _Dopt[ia];
            *Eopt = _Eopt[ia];
            *stack = _stack[ia];
            *sem = smax[ia];
            ssmax = smax[ia];
        }
    }

}

int main(int argc, char *argv[])
{
    int i;

    if (argc != 21) {
        fprintf(stderr, "Usage: %s M0 H0 T0 TAU A0 A1 NA B0 B1 NB "
            "C0 C1 NC D0 D1 ND E0 E1 NE INPUT\n", argv[0]);
        exit(1);
    }

    float m0 = atof(argv[1]);
    float h0 = atof(argv[2]);
    float t0 = atof(argv[3]);
    float tau = strtof(argv[4], NULL);

    /* A, B, C, D, E */
    float p0[5], p1[5];
    int np[5];

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

    /* Construct the aperture structure from the traces, which is a vector
     * containing pointers to traces */
    su_trace_t tr;
    my_su_trace_t mtr;
    my_aperture_t map;

    map.ap_m = 0;
    map.ap_h = 0;
    map.ap_t = tau;

    i = 0;
    while (su_fgettr(fp, &tr)) {
        transformTr(&tr, &mtr);
        map.traces[i] = mtr;
        i++;
    }

    /*-------------------------------------------------------------------------*/

    int n = 5;
    int outSize = 7;
    float out[outSize];

    char *kernelSource = (char *) malloc(MAXSOURCE * sizeof(char));
    
    FILE * file = fopen("kernel.cl", "r");
    if(file == NULL) {
        printf("Error: open the kernel file (kernel.cl)\n");
        exit(1);
    }
    
    // Read kernel code
    size_t source_size = fread(kernelSource, 1, MAXSOURCE, file);
    
    //Device input buffers
    cl_mem d_p0, d_p1, d_np, d_aopt, d_bopt, d_copt, d_dopt, d_eopt, d_stack, d_smax, d_map;
    //Device output buffer
    cl_mem  d_out;
    
    cl_int err;

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
    size_t bytes_np = sizeof(int) * n;
    size_t bytes_map = sizeof(my_aperture_t);
    size_t bytes_opt = sizeof(float) * np[0];
    size_t bytes_out = sizeof(float) * outSize;
    
    //Numero de workitems em cada local work group (local size)
    size_t localSize[3] = {1, 1, 1};
    size_t globalSize[3] = {
        20,
        20,
        20
    };

    // printf("%f\n", ceil((float)np[0] / localSize[0]));
    
    // Bind to platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    if (platformCount == 0) {
        printf("Error, cound not find any OpenCL platforms on the system.\n");
        exit (2);
    }
    
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount,platforms, NULL);
    
    // Find first device that works
    err = 1;
    for (i = 0; i < platformCount && err !=CL_SUCCESS; i++) {
        // Get ID for the device (CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_GPU, ...)
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);

    }
    
    if (err !=CL_SUCCESS) {
        printf("Error, could not find a valid device.");
        exit (3);
    }
    
    err = clGetDeviceInfo(device_id, CL_DEVICE_NAME,MAX_DEVICE_NAME_SIZE, deviceName, NULL);
    printf("Device: %s \n",deviceName);
    
    if (err !=CL_SUCCESS) {
        printf("Error, could not read the info for device.");
        exit (4);
    }
    
    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    
    if (err !=CL_SUCCESS) {
        printf("Error, could not create the context.");
        exit (5);
    }
    
    // Create a command queue
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
            (const char **) & kernelSource,(const size_t *) &source_size, &err);
            
    if (err !=CL_SUCCESS) {
        printf("Error, could not create program with source.");
        exit (6);
    }
            
    // Build the program executable " --disable-multilib "
    err = clBuildProgram(program, 0, NULL, "-I.", NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        cl_int logStatus;
        char* buildLog = NULL;
        size_t buildLogSize = 0;
        logStatus = clGetProgramBuildInfo (program, device_id, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, &buildLogSize);
        buildLog = (char*)malloc(buildLogSize);
        memset(buildLog, 0, buildLogSize);
        logStatus = clGetProgramBuildInfo (program, device_id, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
        printf("%s", buildLog);
        free(buildLog);
        return err;
    } else if (err!=0) {
        printf("Error, could not build program.\n");
        exit (7);
    }
    
    // Create the compute kernel in the program we wish to run
    
    kernel = clCreateKernel(program, "foo", &err);
    
    if (err !=CL_SUCCESS) {
        printf("Error, could not create the kernel.");
        exit (6);
    }
    
    // Create the input and output arrays in device memory for our calculation
    d_map = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_map, NULL, NULL);
    d_p0 = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_p0, NULL, NULL);
    d_p1 = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_p1, NULL, NULL);
    d_np = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_np, NULL, NULL);
    d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes_out, NULL, NULL);
    
    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(queue, d_map, CL_TRUE, 0, bytes_map, &map, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, d_p0, CL_TRUE, 0, bytes_p0, p0, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, d_p1, CL_TRUE, 0, bytes_p1, p1, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, d_np, CL_TRUE, 0, bytes_np, np, 0, NULL, NULL);

    // Set the arguments to our compute kernel
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_map);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_float), &m0);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_float), &h0);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_float), &t0);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_p0);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_p1);
    err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_np);
    err |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &d_out);
    err |= clSetKernelArg(kernel, 8, np[0] * sizeof(cl_float), NULL);//_Aopt
    err |= clSetKernelArg(kernel, 9, np[0] * sizeof(cl_float), NULL);//_Bopt
    err |= clSetKernelArg(kernel, 10, np[0] * sizeof(cl_float), NULL);//_Copt
    err |= clSetKernelArg(kernel, 11, np[0] * sizeof(cl_float), NULL);//_Dopt
    err |= clSetKernelArg(kernel, 12, np[0] * sizeof(cl_float), NULL);//_Eopt
    err |= clSetKernelArg(kernel, 13, np[0] * sizeof(cl_float), NULL);//_stack
    err |= clSetKernelArg(kernel, 14, np[0] * sizeof(cl_float), NULL);//smax
    
    
    if (err !=CL_SUCCESS) {
        printf("Error, could not set kernel args.");
        exit (7);
    }
    
    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, (const size_t *)globalSize,  (const size_t *)localSize, 0, NULL, NULL);
    // Execute the kernel over the entire range of the data set
    if (err !=CL_SUCCESS) {
        printf("Error, could not enqueue commands.");
        printf(": %d\n", err);
        exit (8);
    }
    
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    if (err !=CL_SUCCESS) {
        printf("Error: %d\n", err);
        exit (8);
    }
    
    err |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &d_out);

    if (err !=CL_SUCCESS) {
        printf("Error: %d\n", err);
        exit (8);
    }

    // Read the results from the device
    err |= clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, bytes_out, out, 0, NULL, NULL );

    if (err !=CL_SUCCESS) {
        printf("Error: %d\n", err);
        exit (8);
    }    
    
    for (i=0; i<outSize; i++) 
        printf("OUT[%d]: %f\n", i, out[i]);
    
    /*-------------------------------------------------------------------------*/

    /* Find the best parameter combination */

    // float a, b, c, d, e, sem, stack;
    // compute_max(&map, m0, h0, t0, p0, p1, np, &a, &b, &c, &d, &e, &sem, &stack);

    // printf("A=%g\n", a);
    // printf("B=%g\n", b);
    // printf("C=%g\n", c);
    // printf("D=%g\n", d);
    // printf("E=%g\n", e);
    // printf("Stack=%g\n", stack);
    // printf("Semblance=%g\n", sem);
    // printf("\n");

    return 0;
}
