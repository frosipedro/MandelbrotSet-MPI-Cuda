#ifndef MANDELBROT_KERNEL_H
#define MANDELBROT_KERNEL_H

// Tipos de paletas disponíveis
typedef enum {
    PALETTE_ULTRA = 0,      // Azul/laranja/branco (padrão)
    PALETTE_FIRE,           // Vermelho/amarelo/laranja
    PALETTE_ICE,            // Azul/ciano/branco
    PALETTE_PSYCHEDELIC,    // Cores vibrantes
    PALETTE_RAINBOW,        // Arco-íris
    PALETTE_MONOCHROME,     // Preto e branco
    PALETTE_OCEAN,          // Tons de azul/verde
    PALETTE_SUNSET          // Por do sol
} PaletteType;

#ifdef __cplusplus
extern "C" {
#endif

void launch_mandelbrot_kernel(unsigned char* d_output, int width, int height,
                               double x_min, double x_max,
                               double y_min, double y_max,
                               int max_iter, int y_offset, int global_height,
                               PaletteType palette);

void check_cuda_error(const char* msg);

#ifdef __cplusplus
}
#endif

#endif // MANDELBROT_KERNEL_H