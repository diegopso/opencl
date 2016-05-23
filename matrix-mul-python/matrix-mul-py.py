import numpy
import pyopencl as cl

import os
# os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
# os.environ['PYOPENCL_CTX'] = '0:1'

N = 800

# create host arrays
h_ma = numpy.ones((N * N,), dtype=numpy.float32)
h_mb = numpy.ones((N * N,), dtype=numpy.float32)
h_mc = numpy.zeros((N * N,), dtype=numpy.float32)
lm = numpy.zeros((N,), dtype=numpy.float32)
print(h_ma)
print(h_mb)
print(h_mc)

# create context, queue and program
context = cl.create_some_context()

cpq = cl.command_queue_properties
queue = cl.CommandQueue(context, None, cpq.PROFILING_ENABLE)

choose = raw_input('Execute kernel 1, 2, 3 ou 4? ');

if choose == '1':
    print "kernel1"
    kernelsource = open('mmul-kernel1.c').read()
elif choose == '2':
    print "kernel2"
    kernelsource = open('mmul-kernel2.c').read()
elif choose == '3':
    print "kernel3"
    kernelsource = open('mmul-kernel3.c').read()
else :
    print "kernel4"
    kernelsource = open('mmul-kernel4.c').read()


program = cl.Program(context, kernelsource).build()

# create device buffers
mf = cl.mem_flags
d_a = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_ma)
d_b = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_mb)
d_c = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_mc)
d_x = cl.LocalMemory(N * numpy.dtype("float32").itemsize)

# run kernel
mmul = program.mmul
if choose == '1':
    print "kernel1"
    mmul.set_scalar_arg_dtypes([numpy.uint32, None, None, None])
    mmul(queue, (N, N), None, numpy.uint32(N), d_a, d_b, d_c)
elif choose == '2':
    print "kernel2"
    mmul.set_scalar_arg_dtypes([numpy.uint32, None, None, None])
    mmul(queue, (N, ), None, numpy.uint32(N), d_a, d_b, d_c)
elif choose == '3':
    print "kernel3"
    mmul.set_scalar_arg_dtypes([numpy.uint32, None, None, None])
    mmul(queue, (N, ), None, numpy.uint32(N), d_a, d_b, d_c)
else:
    print "kernel4"
    mmul.set_scalar_arg_dtypes([numpy.uint32, None, None, None, None])
    mmul(queue, (N, ), None, numpy.uint32(N), d_a, d_b, d_c, d_x)
    

# return results
event = cl.enqueue_copy(queue, h_mc, d_c)

event.wait()

elapsed = 1e-9 * (event.profile.end - event.profile.start)
# Compute execution time (using event profiling):
print("Execution time: %g s " % elapsed)

print(h_ma)
print(h_mb)
print(h_mc)
