#include "metrics.h"
#include <stdio.h>
#include <string.h>

void print_separator(char c, int length) {
    for (int i = 0; i < length; i++) printf("%c", c);
    printf("\n");
}

void print_header(const char* title) {
    int length = 70;
    print_separator('=', length);
    int padding = (length - strlen(title) - 2) / 2;
    printf("%*s %s %*s\n", padding, "", title, padding, "");
    print_separator('=', length);
}

void print_metrics(Metrics* m) {
    print_header("MÉTRICAS DE DESEMPENHO - MANDELBROT MPI+CUDA");
    
    printf("\n📊 CONFIGURAÇÃO\n");
    print_separator('-', 70);
    printf("  Processos MPI:              %d\n", m->num_processes);
    printf("  Dimensões da imagem:        %d x %d pixels\n", m->width, m->height);
    printf("  Total de pixels:            %ld\n", (long)m->width * m->height);
    printf("  Iterações máximas:          %d\n", m->max_iterations);
    printf("  Região do plano complexo:   [%.2f, %.2f] x [%.2f, %.2f]\n",
           m->x_min, m->x_max, m->y_min, m->y_max);
    
    printf("\n⏱️  TEMPOS DE EXECUÇÃO\n");
    print_separator('-', 70);
    printf("  Tempo total:                %.3f segundos\n", m->total_time);
    printf("  Tempo de computação GPU:    %.3f segundos (%.1f%%)\n", 
           m->computation_time,
           100.0 * m->computation_time / m->total_time);
    printf("  Tempo de comunicação MPI:   %.3f segundos (%.1f%%)\n", 
           m->communication_time,
           100.0 * m->communication_time / m->total_time);
    printf("  Tempo de I/O (salvar PNG):  %.3f segundos (%.1f%%)\n",
           m->io_time,
           100.0 * m->io_time / m->total_time);
    double overhead = m->total_time - m->computation_time - m->communication_time;
    printf("  Sobrecarga (overhead):      %.3f segundos (%.1f%%)\n",
           overhead,
           100.0 * overhead / m->total_time);
    
    printf("\n🚀 DESEMPENHO\n");
    print_separator('-', 70);
    double total_pixels = (double)m->width * m->height;
    double mpixels_per_sec = (total_pixels / m->computation_time) / 1e6;
    double throughput = (total_pixels * m->max_iterations / m->computation_time) / 1e9;
    
    printf("  Pixels processados/seg:     %.2f Mpixels/s\n", mpixels_per_sec);
    printf("  Taxa de processamento:      %.2f GFlops (estimado)\n", throughput * 10);
    printf("  Pixels por processo:        %ld pixels\n", 
           (long)(m->height / m->num_processes) * m->width);
    
    if (m->num_processes > 1) {
        printf("\n⚡ EFICIÊNCIA DO PARALELISMO\n");
        print_separator('-', 70);
        double efficiency = 100.0 * m->computation_time / (m->total_time * m->num_processes);
        printf("  Eficiência paralela:        %.1f%%\n", efficiency);
        printf("  Speedup comunicação:        %.2fx (ideal: %dx)\n",
               m->computation_time / m->total_time, m->num_processes);
    }
    
    printf("\n💾 USO DE MEMÓRIA\n");
    print_separator('-', 70);
    double memory_mb = (double)(m->width * m->height) / (1024.0 * 1024.0);
    printf("  Memória da imagem:          %.2f MB\n", memory_mb);
    printf("  Memória por processo:       %.2f MB\n", 
           memory_mb / m->num_processes);
    
    print_separator('=', 70);
    printf("\n✅ Execução concluída com sucesso!\n\n");
}