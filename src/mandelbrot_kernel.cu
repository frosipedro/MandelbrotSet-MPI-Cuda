#include "mandelbrot_kernel.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// ============================================================================
// CONSTANTES E CONFIGURAÇÕES
// ============================================================================

#define ESCAPE_RADIUS 256.0        // Raio maior para smooth coloring
#define ESCAPE_RADIUS_SQ 65536.0   // ESCAPE_RADIUS^2
#define LOG2 0.6931471805599453    // ln(2)
#define PI 3.14159265358979323846

// ============================================================================
// FUNÇÕES DE COLORIZAÇÃO (Device)
// ============================================================================

// Converte HSV para RGB (H: 0-360, S: 0-1, V: 0-1)
__device__ void hsv_to_rgb(double h, double s, double v, 
                           unsigned char* r, unsigned char* g, unsigned char* b) {
    double c = v * s;
    double x = c * (1.0 - fabs(fmod(h / 60.0, 2.0) - 1.0));
    double m = v - c;
    
    double r1, g1, b1;
    
    if (h < 60) {
        r1 = c; g1 = x; b1 = 0;
    } else if (h < 120) {
        r1 = x; g1 = c; b1 = 0;
    } else if (h < 180) {
        r1 = 0; g1 = c; b1 = x;
    } else if (h < 240) {
        r1 = 0; g1 = x; b1 = c;
    } else if (h < 300) {
        r1 = x; g1 = 0; b1 = c;
    } else {
        r1 = c; g1 = 0; b1 = x;
    }
    
    *r = (unsigned char)((r1 + m) * 255.0);
    *g = (unsigned char)((g1 + m) * 255.0);
    *b = (unsigned char)((b1 + m) * 255.0);
}

// Interpolação linear entre duas cores
__device__ void lerp_color(double t, 
                           double r1, double g1, double b1,
                           double r2, double g2, double b2,
                           unsigned char* r, unsigned char* g, unsigned char* b) {
    *r = (unsigned char)((r1 * (1.0 - t) + r2 * t) * 255.0);
    *g = (unsigned char)((g1 * (1.0 - t) + g2 * t) * 255.0);
    *b = (unsigned char)((b1 * (1.0 - t) + b2 * t) * 255.0);
}

// ============================================================================
// PALETAS DE CORES
// ============================================================================

// Paleta "Ultra Fractal" estilo - azul/laranja/branco (PADRÃO)
__device__ void palette_ultra(double t, unsigned char* r, unsigned char* g, unsigned char* b) {
    const double colors[5][3] = {
        {0.0, 0.027, 0.392},   // Azul escuro
        {0.125, 0.420, 0.796}, // Azul claro
        {0.929, 1.0, 1.0},     // Branco/ciano
        {1.0, 0.667, 0.0},     // Laranja
        {0.0, 0.008, 0.0}      // Quase preto
    };
    
    t = fmod(t * 5.0, 5.0);
    int idx = (int)t;
    double frac = t - idx;
    int next = (idx + 1) % 5;
    
    frac = frac * frac * (3.0 - 2.0 * frac); // smoothstep
    
    *r = (unsigned char)((colors[idx][0] * (1.0 - frac) + colors[next][0] * frac) * 255.0);
    *g = (unsigned char)((colors[idx][1] * (1.0 - frac) + colors[next][1] * frac) * 255.0);
    *b = (unsigned char)((colors[idx][2] * (1.0 - frac) + colors[next][2] * frac) * 255.0);
}

// Paleta FIRE - vermelho/amarelo/laranja
__device__ void palette_fire(double t, unsigned char* r, unsigned char* g, unsigned char* b) {
    const double colors[5][3] = {
        {0.0, 0.0, 0.0},       // Preto
        {0.5, 0.0, 0.0},       // Vermelho escuro
        {1.0, 0.3, 0.0},       // Laranja
        {1.0, 0.8, 0.0},       // Amarelo
        {1.0, 1.0, 0.9}        // Branco quente
    };
    
    t = fmod(t * 5.0, 5.0);
    int idx = (int)t;
    double frac = t - idx;
    int next = (idx + 1) % 5;
    
    frac = frac * frac * (3.0 - 2.0 * frac);
    
    *r = (unsigned char)((colors[idx][0] * (1.0 - frac) + colors[next][0] * frac) * 255.0);
    *g = (unsigned char)((colors[idx][1] * (1.0 - frac) + colors[next][1] * frac) * 255.0);
    *b = (unsigned char)((colors[idx][2] * (1.0 - frac) + colors[next][2] * frac) * 255.0);
}

// Paleta ICE - azul/ciano/branco
__device__ void palette_ice(double t, unsigned char* r, unsigned char* g, unsigned char* b) {
    const double colors[5][3] = {
        {0.0, 0.0, 0.2},       // Azul muito escuro
        {0.0, 0.2, 0.5},       // Azul escuro
        {0.0, 0.5, 0.8},       // Azul médio
        {0.4, 0.8, 1.0},       // Ciano claro
        {0.9, 0.95, 1.0}       // Branco azulado
    };
    
    t = fmod(t * 5.0, 5.0);
    int idx = (int)t;
    double frac = t - idx;
    int next = (idx + 1) % 5;
    
    frac = frac * frac * (3.0 - 2.0 * frac);
    
    *r = (unsigned char)((colors[idx][0] * (1.0 - frac) + colors[next][0] * frac) * 255.0);
    *g = (unsigned char)((colors[idx][1] * (1.0 - frac) + colors[next][1] * frac) * 255.0);
    *b = (unsigned char)((colors[idx][2] * (1.0 - frac) + colors[next][2] * frac) * 255.0);
}

// Paleta PSYCHEDELIC - cores vibrantes e saturadas
__device__ void palette_psychedelic(double t, unsigned char* r, unsigned char* g, unsigned char* b) {
    // Múltiplas ondas senoidais com frequências diferentes
    double r1 = 0.5 + 0.5 * sin(2.0 * PI * (t * 3.0 + 0.0));
    double g1 = 0.5 + 0.5 * sin(2.0 * PI * (t * 3.0 + 0.33));
    double b1 = 0.5 + 0.5 * sin(2.0 * PI * (t * 3.0 + 0.67));
    
    // Aumenta saturação
    r1 = r1 * r1;
    g1 = g1 * g1;
    b1 = b1 * b1;
    
    *r = (unsigned char)(r1 * 255.0);
    *g = (unsigned char)(g1 * 255.0);
    *b = (unsigned char)(b1 * 255.0);
}

// Paleta RAINBOW - arco-íris completo
__device__ void palette_rainbow(double t, unsigned char* r, unsigned char* g, unsigned char* b) {
    double hue = fmod(t * 360.0 * 2.0, 360.0); // Duas voltas no arco-íris
    hsv_to_rgb(hue, 1.0, 1.0, r, g, b);
}

// Paleta MONOCHROME - preto e branco
__device__ void palette_monochrome(double t, unsigned char* r, unsigned char* g, unsigned char* b) {
    // Usando curva senoidal para transições suaves
    double value = 0.5 + 0.5 * sin(2.0 * PI * t * 4.0);
    value = value * value; // Aumenta contraste
    
    unsigned char v = (unsigned char)(value * 255.0);
    *r = v;
    *g = v;
    *b = v;
}

// Paleta OCEAN - tons de azul e verde
__device__ void palette_ocean(double t, unsigned char* r, unsigned char* g, unsigned char* b) {
    const double colors[6][3] = {
        {0.0, 0.05, 0.15},     // Azul profundo
        {0.0, 0.15, 0.3},      // Azul escuro
        {0.0, 0.3, 0.5},       // Azul médio
        {0.0, 0.5, 0.5},       // Azul-verde
        {0.2, 0.7, 0.6},       // Verde-água
        {0.6, 0.9, 0.85}       // Turquesa claro
    };
    
    t = fmod(t * 6.0, 6.0);
    int idx = (int)t;
    double frac = t - idx;
    int next = (idx + 1) % 6;
    
    frac = frac * frac * (3.0 - 2.0 * frac);
    
    *r = (unsigned char)((colors[idx][0] * (1.0 - frac) + colors[next][0] * frac) * 255.0);
    *g = (unsigned char)((colors[idx][1] * (1.0 - frac) + colors[next][1] * frac) * 255.0);
    *b = (unsigned char)((colors[idx][2] * (1.0 - frac) + colors[next][2] * frac) * 255.0);
}

// Paleta SUNSET - cores de pôr do sol
__device__ void palette_sunset(double t, unsigned char* r, unsigned char* g, unsigned char* b) {
    const double colors[6][3] = {
        {0.1, 0.0, 0.2},       // Roxo escuro
        {0.4, 0.0, 0.4},       // Roxo
        {0.8, 0.2, 0.3},       // Vermelho rosado
        {1.0, 0.5, 0.2},       // Laranja
        {1.0, 0.8, 0.3},       // Amarelo dourado
        {1.0, 0.95, 0.8}       // Amarelo claro
    };
    
    t = fmod(t * 6.0, 6.0);
    int idx = (int)t;
    double frac = t - idx;
    int next = (idx + 1) % 6;
    
    frac = frac * frac * (3.0 - 2.0 * frac);
    
    *r = (unsigned char)((colors[idx][0] * (1.0 - frac) + colors[next][0] * frac) * 255.0);
    *g = (unsigned char)((colors[idx][1] * (1.0 - frac) + colors[next][1] * frac) * 255.0);
    *b = (unsigned char)((colors[idx][2] * (1.0 - frac) + colors[next][2] * frac) * 255.0);
}

// Função que seleciona a paleta correta
__device__ void apply_palette(double t, int palette_type, 
                              unsigned char* r, unsigned char* g, unsigned char* b) {
    switch(palette_type) {
        case 1:  palette_fire(t, r, g, b); break;
        case 2:  palette_ice(t, r, g, b); break;
        case 3:  palette_psychedelic(t, r, g, b); break;
        case 4:  palette_rainbow(t, r, g, b); break;
        case 5:  palette_monochrome(t, r, g, b); break;
        case 6:  palette_ocean(t, r, g, b); break;
        case 7:  palette_sunset(t, r, g, b); break;
        default: palette_ultra(t, r, g, b); break;
    }
}

// ============================================================================
// OTIMIZAÇÕES DE ESCAPE
// ============================================================================

// Verifica se ponto está na cardioide principal 
__device__ bool in_cardioid(double cx, double cy) {
    double cy2 = cy * cy;
    double q = (cx - 0.25) * (cx - 0.25) + cy2;
    return q * (q + (cx - 0.25)) <= 0.25 * cy2;
}

// Verifica se ponto está no bulbo período-2 
__device__ bool in_period2_bulb(double cx, double cy) {
    double temp = cx + 1.0;
    return temp * temp + cy * cy <= 0.0625;  // raio = 0.25, raio^2 = 0.0625
}

// ============================================================================
// CÁLCULO DO MANDELBROT COM SMOOTH ITERATION COUNT
// ============================================================================

__device__ double mandelbrot_smooth(double cx, double cy, int max_iter) {
    // Otimização: escape rápido para regiões conhecidas do conjunto
    if (in_cardioid(cx, cy) || in_period2_bulb(cx, cy)) {
        return (double)max_iter;
    }
    
    double x = 0.0, y = 0.0;
    double x2 = 0.0, y2 = 0.0;
    int iter = 0;
    
    // Loop principal com variáveis pré-calculadas
    while (x2 + y2 <= ESCAPE_RADIUS_SQ && iter < max_iter) {
        y = 2.0 * x * y + cy;
        x = x2 - y2 + cx;
        x2 = x * x;
        y2 = y * y;
        iter++;
    }
    
    if (iter == max_iter) {
        return (double)max_iter;
    }
    
    // Smooth iteration count usando fórmula de escape normalizado
    // Isso elimina as "bandas" de cor e cria gradientes suaves
    double log_zn = log(x2 + y2) / 2.0;
    double nu = log(log_zn / LOG2) / LOG2;
    
    return (double)iter + 1.0 - nu;
}

// Versão com detecção de periodicidade (mais rápida para pontos dentro do conjunto)
__device__ double mandelbrot_smooth_periodic(double cx, double cy, int max_iter) {
    if (in_cardioid(cx, cy) || in_period2_bulb(cx, cy)) {
        return (double)max_iter;
    }
    
    double x = 0.0, y = 0.0;
    double x2 = 0.0, y2 = 0.0;
    
    // Variáveis para detecção de periodicidade
    double x_old = 0.0, y_old = 0.0;
    int check = 3;
    int check_counter = 0;
    
    int iter = 0;
    
    while (x2 + y2 <= ESCAPE_RADIUS_SQ && iter < max_iter) {
        y = 2.0 * x * y + cy;
        x = x2 - y2 + cx;
        x2 = x * x;
        y2 = y * y;
        iter++;
        
        // Detecção de periodicidade (Floyd's cycle detection simplificado)
        if (x == x_old && y == y_old) {
            return (double)max_iter; // Ponto periódico, está no conjunto
        }
        
        check_counter++;
        if (check_counter == check) {
            check_counter = 0;
            x_old = x;
            y_old = y;
            check *= 2;
        }
    }
    
    if (iter == max_iter) {
        return (double)max_iter;
    }
    
    double log_zn = log(x2 + y2) / 2.0;
    double nu = log(log_zn / LOG2) / LOG2;
    
    return (double)iter + 1.0 - nu;
}

// ============================================================================
// KERNEL PRINCIPAL
// ============================================================================

__global__ void mandelbrot_kernel(unsigned char* output, int width, int height,
                                   double x_min, double x_max,
                                   double y_min, double y_max,
                                   int max_iter, int y_offset, int global_height,
                                   int palette_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= width || idy >= height) return;
    
    // Cálculo das coordenadas no plano complexo
    double x_scale = (x_max - x_min) / (double)width;
    double y_scale = (y_max - y_min) / (double)global_height;
    
    double cx = x_min + (double)idx * x_scale;
    double cy = y_min + (double)(idy + y_offset) * y_scale;
    
    // Calcula iteração suave com detecção de periodicidade
    double smooth_iter = mandelbrot_smooth_periodic(cx, cy, max_iter);
    
    // Índice do pixel na saída (3 bytes por pixel: RGB)
    int pixel_idx = (idy * width + idx) * 3;
    
    if (smooth_iter >= (double)max_iter) {
        // Ponto dentro do conjunto: preto
        output[pixel_idx + 0] = 0;
        output[pixel_idx + 1] = 0;
        output[pixel_idx + 2] = 0;
    } else {
        // Normaliza para [0, 1] com escala logarítmica para melhor distribuição
        double t = smooth_iter / (double)max_iter;
        
        // Aplica correção para realçar detalhes
        t = sqrt(t);  // Raiz quadrada expande as cores nas áreas de baixa iteração
        
        // Cicla as cores várias vezes para mais detalhe visual
        t = fmod(t * 8.0, 1.0);
        
        unsigned char r, g, b;
        apply_palette(t, palette_type, &r, &g, &b);
        
        output[pixel_idx + 0] = r;
        output[pixel_idx + 1] = g;
        output[pixel_idx + 2] = b;
    }
}

// Kernel alternativo com colorização HSV (pode ser usado para variações)
__global__ void mandelbrot_kernel_hsv(unsigned char* output, int width, int height,
                                       double x_min, double x_max,
                                       double y_min, double y_max,
                                       int max_iter, int y_offset, int global_height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= width || idy >= height) return;
    
    double x_scale = (x_max - x_min) / (double)width;
    double y_scale = (y_max - y_min) / (double)global_height;
    
    double cx = x_min + (double)idx * x_scale;
    double cy = y_min + (double)(idy + y_offset) * y_scale;
    
    double smooth_iter = mandelbrot_smooth_periodic(cx, cy, max_iter);
    
    int pixel_idx = (idy * width + idx) * 3;
    
    if (smooth_iter >= (double)max_iter) {
        output[pixel_idx + 0] = 0;
        output[pixel_idx + 1] = 0;
        output[pixel_idx + 2] = 0;
    } else {
        // Colorização HSV - hue baseado na iteração
        double hue = fmod(smooth_iter * 3.0, 360.0);
        double saturation = 0.8;
        double value = 1.0;
        
        // Varia a saturação com base na "distância" de escape
        double t = smooth_iter / (double)max_iter;
        value = 0.6 + 0.4 * (1.0 - t);
        
        unsigned char r, g, b;
        hsv_to_rgb(hue, saturation, value, &r, &g, &b);
        
        output[pixel_idx + 0] = r;
        output[pixel_idx + 1] = g;
        output[pixel_idx + 2] = b;
    }
}

// ============================================================================
// FUNÇÃO DE LANÇAMENTO DO KERNEL
// ============================================================================

void launch_mandelbrot_kernel(unsigned char* d_output, int width, int height,
                               double x_min, double x_max,
                               double y_min, double y_max,
                               int max_iter, int y_offset, int global_height,
                               PaletteType palette) {
    // Block size otimizado para GPUs modernas
    // 32x8 geralmente é bom para kernels com muita computação
    dim3 block_size(32, 8);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    mandelbrot_kernel<<<grid_size, block_size>>>(d_output, width, height,
                                                  x_min, x_max, y_min, y_max,
                                                  max_iter, y_offset, global_height,
                                                  (int)palette);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

// ============================================================================
// FUNÇÕES AUXILIARES
// ============================================================================

void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}