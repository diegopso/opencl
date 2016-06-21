################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../my_semblance_options/my_semblance_com_ponteiro.c \
../my_semblance_options/my_semblance_sem_ponteiro.c 

OBJS += \
./my_semblance_options/my_semblance_com_ponteiro.o \
./my_semblance_options/my_semblance_sem_ponteiro.o 

C_DEPS += \
./my_semblance_options/my_semblance_com_ponteiro.d \
./my_semblance_options/my_semblance_sem_ponteiro.d 


# Each subdirectory must supply rules for building sources it contributes
my_semblance_options/%.o: ../my_semblance_options/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


