################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CL_SRCS += \
../kernel.cl 

C_SRCS += \
../my_semblance.c \
../reg.c \
../su.c 

CL_DEPS += \
./kernel.d 

OBJS += \
./kernel.o \
./my_semblance.o \
./reg.o \
./su.o 

C_DEPS += \
./my_semblance.d \
./reg.d \
./su.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cl
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


