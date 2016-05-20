import numpy
import pyopencl as cl

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '0:1'

N = 1000

# create host arrays
h_a = numpy.random.rand(N).astype(numpy.float32)

# create context, queue and program
context = cl.create_some_context()
queue = cl.CommandQueue(context)
kernelsource = open('integrate.cl').read()
program = cl.Program(context, kernelsource).build()

# create device buffers
mf = cl.mem_flags
d_a = cl.Buffer(context, mf.WRITE_ONLY, h_a.nbytes)

# run kernel
integrate = program.integrate
integrate.set_scalar_arg_dtypes([None, numpy.uint32])
integrate(queue, h_a.shape, h_a.shape, d_a, N)

# return results
cl.enqueue_copy(queue, h_a, d_a)

sum = 0.0
for i in range(11):
	sum += h_a[i]

sum /= N

print(sum)
