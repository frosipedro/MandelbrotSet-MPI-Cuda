#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "image_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void save_png(const char* filename, unsigned char* data, int width, int height) {
    // Os dados já vêm em RGB do kernel CUDA, salva diretamente
    if (stbi_write_png(filename, width, height, 3, data, width * 3)) {
        printf("✓ Imagem salva: %s (%dx%d)\n", filename, width, height);
    } else {
        fprintf(stderr, "✗ Erro ao salvar imagem: %s\n", filename);
    }
}