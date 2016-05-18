import numpy
import pyopencl as cl

import os
#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '0:1'

N = 5

# create host arrays
h_a = numpy.random.rand(N).astype(numpy.float32)
h_b = numpy.random.rand(N).astype(numpy.float32)
h_c = numpy.empty_like(h_a)

# create context, queue and program
context = cl.create_some_context()
queue = cl.CommandQueue(context)
kernelsource = open('integrate.cl').read()
program = cl.Program(context, kernelsource).build()

# create device buffers
mf = cl.mem_flags
d_a = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_a)
d_b = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_b)
d_c = cl.Buffer(context, mf.WRITE_ONLY, h_c.nbytes)

# run kernel
integrate = program.integrate
integrate.set_scalar_arg_dtypes([None, None, None, numpy.uint32])
#vadd(queue, (5,), None, d_a, d_b, d_c, N)
integrate(queue, h_a.shape, None, d_a, d_b, d_c, N)

# return results
cl.enqueue_copy(queue, h_c, d_c)

print(h_a)
print(h_b)
print(h_c)
