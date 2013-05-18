################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../ObjectRecognizer.cpp \
../main.cpp \
../utils.cpp 

CU_SRCS += \
../gpuDetectObjs.cu \
../gpu_utils.cu 

CU_DEPS += \
./gpuDetectObjs.d \
./gpu_utils.d 

OBJS += \
./ObjectRecognizer.o \
./gpuDetectObjs.o \
./gpu_utils.o \
./main.o \
./utils.o 

CPP_DEPS += \
./ObjectRecognizer.d \
./main.d \
./utils.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/home/olehp/diploma/gpu-face-detection/gpu_routines -include /usr/local/include/undef_atomics_int128.h -O3 -gencode arch=compute_11,code=sm_11 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -I/home/olehp/diploma/gpu-face-detection/gpu_routines -include /usr/local/include/undef_atomics_int128.h -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/home/olehp/diploma/gpu-face-detection/gpu_routines -include /usr/local/include/undef_atomics_int128.h -O3 -gencode arch=compute_11,code=sm_11 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -I/home/olehp/diploma/gpu-face-detection/gpu_routines -O3 -gencode arch=compute_11,code=compute_11 -gencode arch=compute_11,code=sm_11 -include /usr/local/include/undef_atomics_int128.h  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


