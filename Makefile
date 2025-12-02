# Compiladores
NVCC = nvcc
MPICC = mpicc

# Flags de compilação
NVCC_FLAGS = -arch=sm_89 -O3 -Iinclude
MPICC_FLAGS = -O3 -Wall -Iinclude

# Diretórios
SRC_DIR = src
UTILS_DIR = utils
INCLUDE_DIR = include
OBJ_DIR = obj
BIN_DIR = bin

# Arquivos objeto
CUDA_OBJS = $(OBJ_DIR)/mandelbrot_kernel.o
C_OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/image_utils.o $(OBJ_DIR)/metrics.o

# Executável
TARGET = $(BIN_DIR)/mandelbrot

# Regra padrão
all: directories $(TARGET)

# Cria diretórios necessários
directories:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)

# Compila arquivos CUDA
$(OBJ_DIR)/mandelbrot_kernel.o: $(SRC_DIR)/mandelbrot_kernel.cu $(INCLUDE_DIR)/mandelbrot_kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compila arquivos C com MPI
$(OBJ_DIR)/main.o: main.c $(INCLUDE_DIR)/mandelbrot_kernel.h $(INCLUDE_DIR)/image_utils.h $(INCLUDE_DIR)/metrics.h
	$(MPICC) $(MPICC_FLAGS) -c $< -o $@

$(OBJ_DIR)/image_utils.o: $(UTILS_DIR)/image_utils.c $(INCLUDE_DIR)/image_utils.h
	$(MPICC) $(MPICC_FLAGS) -c $< -o $@

$(OBJ_DIR)/metrics.o: $(UTILS_DIR)/metrics.c $(INCLUDE_DIR)/metrics.h
	$(MPICC) $(MPICC_FLAGS) -c $< -o $@

# Link final
$(TARGET): $(CUDA_OBJS) $(C_OBJS)
	$(MPICC) $(MPICC_FLAGS) -o $@ $^ -L/usr/local/cuda/lib64 -lcudart -lstdc++

# Limpa arquivos compilados
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) mandelbrot.png

# Executa com 1 processo
run1:
	mpirun -np 1 $(TARGET)

# Executa com 2 processos
run2:
	mpirun -np 2 $(TARGET)

# Executa com 4 processos
run4:
	mpirun -np 4 $(TARGET)

# Executa com 8 processos
run8:
	mpirun -np 8 $(TARGET)

# Teste completo com diferentes números de processos
test: all
	@echo "=== Testando com 1 processo ==="
	@mpirun -np 1 $(TARGET)
	@echo "\n=== Testando com 2 processos ==="
	@mpirun -np 2 $(TARGET)
	@echo "\n=== Testando com 4 processos ==="
	@mpirun -np 4 $(TARGET)

.PHONY: all clean run1 run2 run4 run8 test directories