#source directories and paths
SRC_DIR := src
OBJ_DIR := obj
INC_DIR := inc
BIN_DIR := .
INC_PATH := ./$(INC_DIR)

#compiler and flags
CXX 		:= g++
NVCC		:= nvcc

CXXFLAGS	:= -W -Wall -Wextra -std=c++17 -g
#CXXFLAGS	:= -W -Wall -Wextra -std=c++17

#NVCCFLAGS 	:= -g -G --use_fast_math -O2 -arch=sm_30	#debugging, fast math enabled
NVCCFLAGS 	:= --use_fast_math -O2 -arch=sm_30 		#no debugging, fast math enabled
#NVCCFLAGS 	:= -g -G -O2 -arch=sm_30 			#debugging, no fast math

LDFLAGS 	:= -lfftw3 -lm -lcufft
NVCC_LIBS 	:= -lcufft

#compilation variables
EXC 	:= $(BIN_DIR)/main
SRC	:= $(addprefix $(SRC_DIR)/, main.cc particle.cc grid.cc fourier.cc pic.cu)
OBJ	:= $(addprefix $(OBJ_DIR)/, main.o particle.o grid.o fourier.o pic.o)

#declaring phony targets
.PHONY: all clean

all: $(EXC)

$(EXC): $(OBJ)
	$(NVCC) $(LDFLAGS) $^ $(LDLIBS) -o $@ -I$(INC_PATH)

$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cc $(INC_DIR)/params.h $(INC_DIR)/particle.h $(INC_DIR)/grid.h
	$(CXX) -c $< -o $@ -I$(INC_PATH)

$(OBJ_DIR)/particle.o: $(SRC_DIR)/particle.cc $(INC_DIR)/params.h $(INC_DIR)/particle.h
	$(CXX) $(CXXFLAGS) -c $< -o $@ -I$(INC_PATH)

$(OBJ_DIR)/grid.o: $(SRC_DIR)/grid.cc $(INC_DIR)/grid.h
	$(CXX) $(CXXFLAGS) -c $< -o $@ -I$(INC_PATH)

$(OBJ_DIR)/fourier.o: $(SRC_DIR)/fourier.cc $(INC_DIR)/fourier.h
	$(CXX) $(CXXFLAGS) -c $< -o $@ -I$(INC_PATH)

$(OBJ_DIR)/pic.o: $(SRC_DIR)/pic.cu $(INC_DIR)/pic.h $(INC_DIR)/params.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ $(NVCCLIBS) -I$(INC_PATH)

clean:
	$(RM) $(EXC) $(OBJ)
