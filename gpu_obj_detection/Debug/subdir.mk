################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../AdaBoost.cpp \
../CascadeTrainer.cpp \
../DataSet.cpp \
../DecisionStump.cpp \
../Feature.cpp \
../utils.cpp 

CU_SRCS += \
../GpuDecisionStump.cu \
../gpu_prepare_data_set.cu \
../gpu_utils.cu 

CU_DEPS += \
./GpuDecisionStump.d \
./gpu_prepare_data_set.d \
./gpu_utils.d 

OBJS += \
./AdaBoost.o \
./CascadeTrainer.o \
./DataSet.o \
./DecisionStump.o \
./Feature.o \
./GpuDecisionStump.o \
./gpu_prepare_data_set.o \
./gpu_utils.o \
./utils.o 

CPP_DEPS += \
./AdaBoost.d \
./CascadeTrainer.d \
./DataSet.d \
./DecisionStump.d \
./Feature.d \
./utils.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -include /usr/local/include/undef_atomics_int128.h -O3 -Xcompiler -fPIC -gencode arch=compute_10,code=sm_10 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -include /usr/local/include/undef_atomics_int128.h -O3 -Xcompiler -fPIC --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -include /usr/local/include/undef_atomics_int128.h -O3 -Xcompiler -fPIC -gencode arch=compute_10,code=sm_10 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -O3 -Xcompiler -fPIC -gencode arch=compute_10,code=compute_10 -gencode arch=compute_10,code=sm_10 -include /usr/local/include/undef_atomics_int128.h  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


