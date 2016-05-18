#include <iostream>
#include <CL/cl.hpp>

int init(float* A, int n) {
	int i,j;
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			A[i*n+j] = 1;
		}
	}

	return 0;
}
 
int main(){
	int n = 5;
	int bytes = sizeof(float)*n*n;

    //get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
 
    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device=all_devices[0];
    std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
 
 
    cl::Context context({default_device});
 
    cl::Program::Sources sources;

    // kernel calculates for each element C=A+B
    std::string kernel_code =
	"__kernel void mmul(const int N, __global float *A, __global float  *B,  __global float *C) {"
	"	int	k;"
	"	int	i=get_global_id(0);"
	"	int	j=get_global_id(1);"
	"	int 	tmp=0;"

	"	for(k=0;k<N;k++) {"
	"		tmp += A[i*N+k]*B[k*N+j];"
	"	}"

	"	C[i*N+j]=tmp;"
	"}";
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
 
    cl::Program program(context,sources);
    if(program.build({default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
        exit(1);
    }
      
    // create buffers on the device
    cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,bytes);
    cl::Buffer buffer_B(context,CL_MEM_READ_WRITE,bytes);
    cl::Buffer buffer_C(context,CL_MEM_READ_WRITE,bytes);
 
    float A[n*n],B[n*n];
	init(A, n);
	init(B, n);
 
    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context,default_device);

    //write arrays A and B to the device
    queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,bytes,A);
    queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,bytes,B);
 
    //run the kernel
    cl::KernelFunctor mmul(cl::Kernel(program,"mmul"),queue,cl::NullRange,cl::NDRange(n,n),cl::NullRange);
	mmul(n, buffer_A, buffer_B, buffer_C);
 
    //alternative way to run the kernel
    //cl::Kernel mmul=cl::Kernel(program, "mmul");
	//mmul.setArg(0,n);
    //mmul.setArg(1,buffer_A);
    //mmul.setArg(2,buffer_B);
    //mmul.setArg(3,buffer_C);
    //queue.enqueueNDRangeKernel(mmul, cl::NullRange, cl::NDRange(3, 3), cl::NullRange);
    //queue.finish();
 
    float C[n*n];
    //read result C from the device to array C
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, bytes, C);
 
    std::cout<<" result: \n";
    for(int i=0;i<9;i++){
        std::cout<<C[i]<<" ";
    }

	std::cout<<"\n";

    return 0;
}
