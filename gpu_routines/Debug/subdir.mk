################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../gpu_add.cu \
../gpu_compute_ii.cu \
../gpu_utils.cu 

CU_DEPS += \
./gpu_add.d \
./gpu_compute_ii.d \
./gpu_utils.d 

OBJS += \
./gpu_add.o \
./gpu_compute_ii.o \
./gpu_utils.o 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -include /usr/local/include/undef_atomics_int128.h -G -g -O0 -Xcompiler -fPIC -gencode arch=compute_11,code=sm_11 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -O0 -Xcompiler -fPIC -g -gencode arch=compute_11,code=compute_11 -gencode arch=compute_11,code=sm_11 -include /usr/local/include/undef_atomics_int128.h  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


