################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Application.cpp \
../CpuObjDetector.cpp \
../ImgMedia.cpp \
../WebCamMedia.cpp \
../main.cpp \
../utils.cpp 

CU_SRCS += \
../GpuObjDetector.cu 

CU_DEPS += \
./GpuObjDetector.d 

OBJS += \
./Application.o \
./CpuObjDetector.o \
./GpuObjDetector.o \
./ImgMedia.o \
./WebCamMedia.o \
./main.o \
./utils.o 

CPP_DEPS += \
./Application.d \
./CpuObjDetector.d \
./ImgMedia.d \
./WebCamMedia.d \
./main.d \
./utils.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/home/olehp/diploma/gpu-face-detection/gpu_routines -include /usr/local/include/undef_atomics_int128.h -O3 -m32 -gencode arch=compute_11,code=sm_11 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -I/home/olehp/diploma/gpu-face-detection/gpu_routines -include /usr/local/include/undef_atomics_int128.h -O3 -m32 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/home/olehp/diploma/gpu-face-detection/gpu_routines -include /usr/local/include/undef_atomics_int128.h -O3 -m32 -gencode arch=compute_11,code=sm_11 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -I/home/olehp/diploma/gpu-face-detection/gpu_routines -O3 -gencode arch=compute_11,code=compute_11 -gencode arch=compute_11,code=sm_11 -include /usr/local/include/undef_atomics_int128.h -m32  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


