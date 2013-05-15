################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../main.cpp 

CC_SRCS += \
../ada_boost_test.cc \
../decision_stump_test.cc \
../feature_test.cc \
../get_mat_col_test.cc \
../load_images_test.cc \
../matrix_equal_test.cc \
../normalize_mat_test.cc \
../opencv_test.cc \
../prepare_data_set_test.cc \
../sqr_test.cc \
../test_utils.cc \
../to_ii_test.cc 

OBJS += \
./ada_boost_test.o \
./decision_stump_test.o \
./feature_test.o \
./get_mat_col_test.o \
./load_images_test.o \
./main.o \
./matrix_equal_test.o \
./normalize_mat_test.o \
./opencv_test.o \
./prepare_data_set_test.o \
./sqr_test.o \
./test_utils.o \
./to_ii_test.o 

CC_DEPS += \
./ada_boost_test.d \
./decision_stump_test.d \
./feature_test.d \
./get_mat_col_test.d \
./load_images_test.d \
./matrix_equal_test.d \
./normalize_mat_test.d \
./opencv_test.d \
./prepare_data_set_test.d \
./sqr_test.d \
./test_utils.d \
./to_ii_test.d 

CPP_DEPS += \
./main.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I"/home/olehp/diploma/gpu-face-detection/gpu_obj_detection" -O0 -g3 -Wall -c -fmessage-length=0 --std=c++0x -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I"/home/olehp/diploma/gpu-face-detection/gpu_obj_detection" -O0 -g3 -Wall -c -fmessage-length=0 --std=c++0x -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


