cmake_minimum_required(VERSION 2.6)
project(reg)


SET(OPENCL_DIRECTORY "/opt/altera/14.1/hld/host/include/")

SET(OPENCL_FLAG "-lm -lOpenCl")

add_definitions(-std=gnu99
    -D_FILE_OFFSET_BITS=64
    -D_LARGEFILE64_SOURCE
	${OPENCL_FLAG})

FIND_LIBRARY(OPENCL_LIBRARIES OpenCL
  ENV LD_LIBRARY_PATH
)
 
GET_FILENAME_COMPONENT(OPENCL_LIB_DIR ${OPENCL_LIBRARIES} PATH)
GET_FILENAME_COMPONENT(_OPENCL_INC_CAND ${OPENCL_LIB_DIR}/../../include ABSOLUTE)
 
FIND_PATH(OPENCL_INCLUDE_DIRS CL/cl.h PATHS ${_OPENCL_INC_CAND} "/usr/local/cuda/include")
FIND_PATH(_OPENCL_CPP_INCLUDE_DIRS CL/cl.hpp PATHS ${_OPENCL_INC_CAND} "/usr/local/cuda/include")
 
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
 
include_directories( ${OPENCL_INCLUDE_DIRS} )
include_directories( "${CMAKE_SOURCE_DIR}/khronos" )



include_directories(${CMAKE_SOURCE_DIR})

add_executable(reg
    reg.c
    semblance.c
    su.c)

target_link_libraries(reg m ${OPENCL_LIBRARIES})
