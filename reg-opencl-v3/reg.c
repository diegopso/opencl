#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#include <utils.h>
#include <vector.h>
#include <semblance.h>
#include <my_semblance.h>

#include <CL/opencl.h>

#include <errno.h>
#include <string.h>
#include <unistd.h>

#include <su.h>

#define MAXSOURCE 8192
#define MAX_DEVICE_NAME_SIZE 100
#define LOCALSIZE 8

#define DATA_SIZE 2502

void checkError(cl_int err, const char *operation);

/*Transform aperture_t to my_aperture_t */
my_aperture_t transform(aperture_t ap) {
	my_aperture_t my_ap;

	/* copy ap_t value */
	my_ap.ap_t = ap.ap_t;

	/* copy tr value */
	for (int i = 0; i < ap.traces.len; i++) {

		su_trace_t *tr = vector_get(ap.traces, i);
		my_su_trace_t my_tr;

		/* copy tr data value */
		for (int j = 0; j < DATA_SIZE; j++) {
			my_tr.data[j] = tr->data[j];
		}

		my_tr.dt = tr->dt;
		my_tr.ns = tr->ns;
		my_tr.gy = tr->gy;
		my_tr.sy = tr->sy;

		my_ap.traces[i] = my_tr;
	}
	my_ap.traces_len = ap.traces.len;

	return my_ap;
}

int main(int argc, char *argv[]) {
	int i;

	int n = 5;
	int outSize = 7*20;
	/* A, B, C, D, E */
	float p0[n], p1[n];
	int np[n];
	float out[outSize];

	if (argc != 22) {
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
		p0[i] = atof(argv[5 + 3 * i]);
		p1[i] = atof(argv[5 + 3 * i + 1]);
		np[i] = atoi(argv[5 + 3 * i + 2]);
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

	my_aperture_t my_ap = transform(ap);

	/*-------------------------------------------------------------------------*/

	int device_type = atof(argv[21]);

	char *kernelSource = (char *) malloc(MAXSOURCE * sizeof(char));

	FILE * file = fopen("kernel.cl", "r");
	if(file == NULL) {
		printf("Error: open the kernel file (kernel.cl)\n");
		exit(1);
	}

	// Read kernel code
	size_t source_size = fread(kernelSource, 1, MAXSOURCE, file);

	//Device input buffers
	cl_mem d_my_ap;
	cl_mem d_p0,
	d_p1, d_np, d_aopt, d_bopt, d_copt, d_dopt, d_eopt, d_stack, d_smax;
	//Device output buffer
	cl_mem d_out;

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
	size_t bytes_my_ap = sizeof(my_aperture_t);
	size_t bytes_p0 = sizeof(float) * n;
	size_t bytes_p1 = sizeof(float) * n;
	size_t bytes_np = sizeof(int) * n;
	size_t bytes_opt = sizeof(float) * np[0];
	size_t bytes_out = sizeof(float) * outSize;

	//Numero de workitems em cada local work group (local size)
	//    size_t localSize[3] = {LOCALSIZE, LOCALSIZE, LOCALSIZE};
	//
	//    size_t globalSize[3] = {
	//        ceil((float)np[0] / (float)localSize[0]),
	//        ceil((float)np[1] / (float)localSize[1]),
	//        ceil((float)np[2] / (float)localSize[2])
	//    };

	// Bind to platforms
	clGetPlatformIDs(0, NULL, &platformCount);
	if (platformCount == 0) {
		printf("Error, cound not find any OpenCL platforms on the system.\n");
		exit(2);
	}

	platforms = (cl_platform_id*) malloc(
			sizeof(cl_platform_id) * platformCount);
	clGetPlatformIDs(platformCount, platforms, NULL);

	// Find first device that works
	err = 1;

	cl_device_type device_type_opencl;

	if(device_type == 0) {
		device_type_opencl = CL_DEVICE_TYPE_CPU;
	} else {
		device_type_opencl = CL_DEVICE_TYPE_GPU;
	}

	for (i = 0; i < platformCount && err != CL_SUCCESS; i++) {
		// Get ID for the device (CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_GPU, ...)
		err = clGetDeviceIDs(platforms[i], device_type_opencl, 1, &device_id,
				NULL);

	}

	checkError(err, "get device");

	if (err != CL_SUCCESS) {
		printf("Error, could not find a valid device.");
		exit(3);
	}

	err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, MAX_DEVICE_NAME_SIZE,
			deviceName, NULL);
	printf("Device: %s \n", deviceName);

	if (err != CL_SUCCESS) {
		printf("Error, could not read the info for device.");
		exit(4);
	}

	// Create a context
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

	if (err != CL_SUCCESS) {
		printf("Error, could not create the context.");
		exit(5);
	}

	if(device_type == 0) {
		// Create a command queue
#ifdef CL_VERSION_2_0
		cl_queue_properties queue_properties[] = {
				CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
				0};
		queue = clCreateCommandQueueWithProperties(context, device_id,
				queue_properties , &err);
#else
		queue = clCreateCommandQueue(context, device_id,
				CL_QUEUE_PROFILING_ENABLE, &err);
#endif


	} else {

		queue = clCreateCommandQueue(context, device_id,
				CL_QUEUE_PROFILING_ENABLE, &err);
	}

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1,
			(const char **) &kernelSource, (const size_t *) &source_size, &err);

	if (err != CL_SUCCESS) {
		printf("Error, could not create program with source.");
		exit(6);
	}

	// Build the program executable " --disable-multilib "
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err == CL_BUILD_PROGRAM_FAILURE) {
		cl_int logStatus;
		char* buildLog = NULL;
		size_t buildLogSize = 0;
		logStatus = clGetProgramBuildInfo(program, device_id,
				CL_PROGRAM_BUILD_LOG, buildLogSize, NULL, &buildLogSize);
		buildLog = (char*) malloc(buildLogSize);
		memset(buildLog, 0, buildLogSize);
		logStatus = clGetProgramBuildInfo(program, device_id,
				CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
		printf("ERROR %d (logsz = %d): [[%s]]\n", err, buildLogSize, buildLog);
		free(buildLog);
		return err;
	} else if (err != 0) {
		printf("Error, could not build program.\n");
		exit(7);
	}

	// Create the compute kernel in the program we wish to run

	kernel = clCreateKernel(program, "calculate", &err);
	checkError(err, "Error, could not create the kernel.");

	float smax[20];
	for (int i = 0; i < np[0]; i++) {
		smax[i] = -1.0;
	}
	size_t bytes_smax = sizeof(float) * np[0];

	// Create the input and output arrays in device memory for our calculation
	d_my_ap = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_my_ap, NULL, NULL);
	d_p0 = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_p0, NULL, NULL);
	d_p1 = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_p1, NULL, NULL);
	d_np = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_np, NULL, NULL);
	d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes_out, NULL, NULL);

	d_aopt = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes_smax, NULL, NULL);
	d_bopt = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes_smax, NULL, NULL);
	d_copt = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes_smax, NULL, NULL);
	d_dopt = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes_smax, NULL, NULL);
	d_eopt = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes_smax, NULL, NULL);
	d_stack = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes_smax, NULL,
			NULL);
	d_smax = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_smax, NULL, NULL);

	// Write our data set into the input array in device memory
	err = clEnqueueWriteBuffer(queue, d_my_ap, CL_TRUE, 0, bytes_my_ap,
			(const void*) &my_ap, 0, NULL, NULL);

	err |= clEnqueueWriteBuffer(queue, d_p0, CL_TRUE, 0, bytes_p0, p0, 0, NULL,
			NULL);
	err |= clEnqueueWriteBuffer(queue, d_p1, CL_TRUE, 0, bytes_p1, p1, 0, NULL,
			NULL);
	err |= clEnqueueWriteBuffer(queue, d_np, CL_TRUE, 0, bytes_np, np, 0, NULL,
			NULL);
	err |= clEnqueueWriteBuffer(queue, d_smax, CL_FALSE, 0, bytes_smax, smax, 0,
			NULL, NULL);
	checkError(err, "clEnqueueWriteBuffer");

	//	err = clWaitForEvents(1, &event1);

	// Set the arguments to our compute kernel

	err = clSetKernelArg(kernel, 0, sizeof(d_my_ap), &d_my_ap);
	checkError(err, "setarg0");
	err = clSetKernelArg(kernel, 11, sizeof(float), &m0);
	checkError(err, "setarg1");
	err = clSetKernelArg(kernel, 12, sizeof(float), &h0);
	checkError(err, "setarg2");
	err = clSetKernelArg(kernel, 13, sizeof(float), &t0);
	checkError(err, "setarg3");
	err = clSetKernelArg(kernel, 1, sizeof(d_p0), &d_p0);
	checkError(err, "setarg4");
	err = clSetKernelArg(kernel, 2, sizeof(d_p1), &d_p1);
	checkError(err, "setarg5");
	err = clSetKernelArg(kernel, 3, sizeof(d_np), &d_np);
	checkError(err, "setarg6");
	err = clSetKernelArg(kernel, 10, sizeof(d_out), &d_out);
	checkError(err, "setarg7");
	err = clSetKernelArg(kernel, 4, sizeof(d_aopt), &d_aopt); //_Aopt
	checkError(err, "setarg8");
	err = clSetKernelArg(kernel, 5, sizeof(d_bopt), &d_bopt); //_Bopt
	checkError(err, "setarg9");
	err = clSetKernelArg(kernel, 6, sizeof(d_copt), &d_copt); //_Copt
	checkError(err, "setarg10");
	err = clSetKernelArg(kernel, 7, sizeof(d_dopt), &d_dopt); //_Dopt
	checkError(err, "setarg11");
	err = clSetKernelArg(kernel, 8, sizeof(d_eopt), &d_eopt); //_Eopt
	checkError(err, "setarg12");
	err = clSetKernelArg(kernel, 9, sizeof(d_stack), &d_stack); //_stack
	checkError(err, "setarg13");
	err = clSetKernelArg(kernel, 14, sizeof(d_smax), &d_smax); //smax
	checkError(err, "setarg14");

//	for (int k = 0; k < 10; ++k) {
		size_t localSize[3] = {2, 2, 2};

		size_t globalSize[3] = { 20, 20, 20 };

		cl_event event;
		err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
				(const size_t *) globalSize, (const size_t *) localSize, 0, NULL,
				&event);

		// Execute the kernel over the entire range of the data set

		checkError(err, "Error, could not enqueue commands.");

		clFlush(queue);

		// Wait for the command queue to get serviced before reading back results
		clFinish(queue);

		// Read the results from the device
		clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, bytes_out, out, 0, NULL,
				NULL);

		//profiling
		//clWaitForEvents(1, &event);

		cl_ulong time_start, time_end;
		double total_time;

		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
				sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end),
				&time_end, NULL);
		total_time = time_end - time_start;
		printf("%0.3f\n", (total_time / 1000000.0));
//	}
//
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseMemObject(d_my_ap);
	clReleaseMemObject(d_p0);
	clReleaseMemObject(d_p1);
	clReleaseMemObject(d_np);
	clReleaseMemObject(d_out);
	clReleaseMemObject(d_aopt);
	clReleaseMemObject(d_bopt);
	clReleaseMemObject(d_copt);
	clReleaseMemObject(d_dopt);
	clReleaseMemObject(d_eopt);
	clReleaseMemObject(d_stack);
	clReleaseMemObject(d_smax);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	/*-------------------------------------------------------------------------*/
 /*
	printf("A=%g\n", out[0]);
			printf("B=%g\n", out[1]);
			printf("C=%g\n", out[2]);
			printf("D=%g\n", out[3]);
			printf("E=%g\n", out[4]);
			printf("Stack=%g\n", out[5]);
			printf("Semblance=%g\n", out[6]);
*/
	printf("--------------------\n");
		printf("%g\n", out[0]);
		printf("%g\n", out[1]);
		printf("%g\n", out[2]);
		printf("%g\n", out[3]);
		printf("%g\n", out[4]);
		printf("%g\n", out[5]);
		printf("%g\n", out[6]);
		printf("\n");

	return 0;
}

void checkError(cl_int err, const char *operation) {
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Error during operation '%s': %d\n", operation, err);
		exit(1);
	}
}
