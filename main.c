#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "mandelbrot_kernel.h"
#include "image_utils.h"
#include "metrics.h"

// Configurações padrão do Mandelbrot
#define DEFAULT_WIDTH 12288
#define DEFAULT_HEIGHT 12288
#define DEFAULT_MAX_ITER 2000
#define DEFAULT_X_MIN -2.5
#define DEFAULT_X_MAX 1.0
#define DEFAULT_Y_MIN -1.25
#define DEFAULT_Y_MAX 1.25

void print_usage(const char* program_name) {
    printf("\nUso: mpirun -np <processos> %s [opções]\n\n", program_name);
    printf("Opções:\n");
    printf("  -w <width>      Largura da imagem (padrão: %d)\n", DEFAULT_WIDTH);
    printf("  -h <height>     Altura da imagem (padrão: %d)\n", DEFAULT_HEIGHT);
    printf("  -i <iter>       Iterações máximas (padrão: %d)\n", DEFAULT_MAX_ITER);
    printf("  -xmin <valor>   Limite esquerdo (padrão: %.2f)\n", DEFAULT_X_MIN);
    printf("  -xmax <valor>   Limite direito (padrão: %.2f)\n", DEFAULT_X_MAX);
    printf("  -ymin <valor>   Limite inferior (padrão: %.2f)\n", DEFAULT_Y_MIN);
    printf("  -ymax <valor>   Limite superior (padrão: %.2f)\n", DEFAULT_Y_MAX);
    printf("  -o <arquivo>    Nome do arquivo de saída (padrão: mandelbrot.png)\n");
    printf("  -p <paleta>     Paleta de cores (padrão: ultra)\n");
    printf("  --help          Mostra esta mensagem\n\n");
    printf("Paletas disponíveis:\n");
    printf("  ultra        - Azul/laranja/branco (clássica)\n");
    printf("  fire         - Vermelho/amarelo/laranja (fogo)\n");
    printf("  ice          - Azul/ciano/branco (gelo)\n");
    printf("  psychedelic  - Cores vibrantes e saturadas\n");
    printf("  rainbow      - Arco-íris completo\n");
    printf("  monochrome   - Preto e branco\n");
    printf("  ocean        - Tons de azul e verde\n");
    printf("  sunset       - Cores de pôr do sol\n\n");
    printf("Exemplos de regiões interessantes:\n");
    printf("  Zoom 1 (espiral): -xmin -0.8 -xmax -0.4 -ymin -0.2 -ymax 0.2 -i 3000\n");
    printf("  Zoom 2 (seahorse valley): -xmin -0.75 -xmax -0.73 -ymin 0.1 -ymax 0.12 -i 5000\n");
    printf("  Zoom 3 (elefante): -xmin 0.28 -xmax 0.30 -ymin 0.008 -ymax 0.012 -i 4000\n\n");
}

PaletteType parse_palette(const char* name) {
    if (strcmp(name, "fire") == 0) return PALETTE_FIRE;
    if (strcmp(name, "ice") == 0) return PALETTE_ICE;
    if (strcmp(name, "psychedelic") == 0) return PALETTE_PSYCHEDELIC;
    if (strcmp(name, "rainbow") == 0) return PALETTE_RAINBOW;
    if (strcmp(name, "monochrome") == 0) return PALETTE_MONOCHROME;
    if (strcmp(name, "ocean") == 0) return PALETTE_OCEAN;
    if (strcmp(name, "sunset") == 0) return PALETTE_SUNSET;
    return PALETTE_ULTRA; // padrão
}

const char* palette_name(PaletteType p) {
    switch(p) {
        case PALETTE_FIRE: return "fire";
        case PALETTE_ICE: return "ice";
        case PALETTE_PSYCHEDELIC: return "psychedelic";
        case PALETTE_RAINBOW: return "rainbow";
        case PALETTE_MONOCHROME: return "monochrome";
        case PALETTE_OCEAN: return "ocean";
        case PALETTE_SUNSET: return "sunset";
        default: return "ultra";
    }
}

int main(int argc, char** argv) {
    int rank, size;
    double start_time, end_time, comp_start, comp_end, comm_time = 0.0;
    
    // Parâmetros configuráveis
    int WIDTH = DEFAULT_WIDTH;
    int HEIGHT = DEFAULT_HEIGHT;
    int MAX_ITER = DEFAULT_MAX_ITER;
    double X_MIN = DEFAULT_X_MIN;
    double X_MAX = DEFAULT_X_MAX;
    double Y_MIN = DEFAULT_Y_MIN;
    double Y_MAX = DEFAULT_Y_MAX;
    char output_file[256] = "mandelbrot.png";
    PaletteType palette = PALETTE_ULTRA;
    
    // Verifica se --help foi passado antes de inicializar MPI
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    // Inicializa MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Parse dos argumentos (apenas o rank 0 faz o parse)
    if (rank == 0) {
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
                WIDTH = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-h") == 0 && i + 1 < argc) {
                HEIGHT = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                MAX_ITER = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-xmin") == 0 && i + 1 < argc) {
                X_MIN = atof(argv[++i]);
            } else if (strcmp(argv[i], "-xmax") == 0 && i + 1 < argc) {
                X_MAX = atof(argv[++i]);
            } else if (strcmp(argv[i], "-ymin") == 0 && i + 1 < argc) {
                Y_MIN = atof(argv[++i]);
            } else if (strcmp(argv[i], "-ymax") == 0 && i + 1 < argc) {
                Y_MAX = atof(argv[++i]);
            } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
                strncpy(output_file, argv[++i], sizeof(output_file) - 1);
            } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
                palette = parse_palette(argv[++i]);
            }
        }
    }
    
    // Broadcast dos parâmetros para todos os processos
    MPI_Bcast(&WIDTH, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&HEIGHT, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&MAX_ITER, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&X_MIN, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&X_MAX, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Y_MIN, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Y_MAX, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(output_file, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&palette, sizeof(PaletteType), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    start_time = MPI_Wtime();
    
    // Calcula a porção de cada processo
    int rows_per_process = HEIGHT / size;
    int y_start = rank * rows_per_process;
    int local_height = (rank == size - 1) ? HEIGHT - y_start : rows_per_process;
    
    if (rank == 0) {
        printf("\n🚀 Iniciando cálculo do Mandelbrot Set...\n");
        printf("   Processos MPI: %d\n", size);
        printf("   Resolução: %dx%d\n", WIDTH, HEIGHT);
        printf("   Iterações: %d\n", MAX_ITER);
        printf("   Paleta: %s\n", palette_name(palette));
        printf("   Região: [%.4f, %.4f] x [%.4f, %.4f]\n\n", X_MIN, X_MAX, Y_MIN, Y_MAX);
    }
    
    // Aloca memória no host para a porção local (RGB = 3 bytes por pixel)
    unsigned char* h_local = (unsigned char*)malloc(WIDTH * local_height * 3);
    
    // Aloca memória na GPU (RGB = 3 bytes por pixel)
    unsigned char* d_output;
    cudaMalloc((void**)&d_output, WIDTH * local_height * 3);
    check_cuda_error("malloc");
    
    // Barreira para sincronizar antes de medir tempo
    MPI_Barrier(MPI_COMM_WORLD);
    comp_start = MPI_Wtime();
    
    // Executa o kernel CUDA - passa os limites GLOBAIS, não locais!
    launch_mandelbrot_kernel(d_output, WIDTH, local_height,
                             X_MIN, X_MAX, Y_MIN, Y_MAX,
                             MAX_ITER, y_start, HEIGHT, palette);
    cudaDeviceSynchronize();
    check_cuda_error("kernel execution");
    
    // Copia resultado da GPU para o host (RGB = 3 bytes por pixel)
    cudaMemcpy(h_local, d_output, WIDTH * local_height * 3, cudaMemcpyDeviceToHost);
    check_cuda_error("memcpy D2H");
    
    comp_end = MPI_Wtime();
    
    // Libera memória da GPU
    cudaFree(d_output);
    
    // Processo 0 coleta todos os resultados (RGB = 3 bytes por pixel)
    unsigned char* h_full = NULL;
    if (rank == 0) {
        h_full = (unsigned char*)malloc(WIDTH * HEIGHT * 3);
    }
    
    double comm_start = MPI_Wtime();
    
    // Prepara arrays para MPI_Gatherv
    int* recvcounts = NULL;
    int* displs = NULL;
    
    if (rank == 0) {
        recvcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        
        for (int i = 0; i < size; i++) {
            int proc_y_start = i * rows_per_process;
            int proc_height = (i == size - 1) ? HEIGHT - proc_y_start : rows_per_process;
            recvcounts[i] = WIDTH * proc_height * 3;  // RGB = 3 bytes por pixel
            displs[i] = proc_y_start * WIDTH * 3;
        }
    }
    
    MPI_Gatherv(h_local, WIDTH * local_height * 3, MPI_UNSIGNED_CHAR,
                h_full, recvcounts, displs, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);
    
    double comm_end = MPI_Wtime();
    comm_time = comm_end - comm_start;
    
    // Processo 0 salva a imagem e exibe métricas
    if (rank == 0) {
        end_time = MPI_Wtime();
        
        double io_start = MPI_Wtime();
        save_png(output_file, h_full, WIDTH, HEIGHT);
        double io_end = MPI_Wtime();
        
        // Preenche estrutura de métricas
        Metrics metrics;
        metrics.num_processes = size;
        metrics.width = WIDTH;
        metrics.height = HEIGHT;
        metrics.max_iterations = MAX_ITER;
        metrics.x_min = X_MIN;
        metrics.x_max = X_MAX;
        metrics.y_min = Y_MIN;
        metrics.y_max = Y_MAX;
        metrics.total_time = end_time - start_time;
        metrics.computation_time = comp_end - comp_start;
        metrics.communication_time = comm_time;
        metrics.io_time = io_end - io_start;
        
        print_metrics(&metrics);
        
        free(h_full);
        free(recvcounts);
        free(displs);
    }
    
    free(h_local);
    
    MPI_Finalize();
    return 0;
}