#!/bin/bash

# Script para testar diferentes regiões do Mandelbrot

echo "======================================"
echo "  Testando Regiões do Mandelbrot"
echo "======================================"

# Número de processos MPI (ajuste conforme sua máquina)
NP=4

# 1. Visão completa (padrão)
echo -e "\n[1/6] Gerando visão completa..."
mpirun -np $NP ./bin/mandelbrot -w 8192 -h 8192 -i 1500 -o mandelbrot_full.png

# 2. Zoom na espiral (região colorida)
echo -e "\n[2/6] Gerando zoom na espiral..."
mpirun -np $NP ./bin/mandelbrot -w 8192 -h 8192 -i 3000 \
    -xmin -0.8 -xmax -0.4 -ymin -0.2 -ymax 0.2 \
    -o mandelbrot_spiral.png

# 3. Seahorse Valley (detalhes incríveis)
echo -e "\n[3/6] Gerando Seahorse Valley..."
mpirun -np $NP ./bin/mandelbrot -w 8192 -h 8192 -i 5000 \
    -xmin -0.747 -xmax -0.737 -ymin 0.095 -ymax 0.105 \
    -o mandelbrot_seahorse.png

# 4. Elephant Valley (formas orgânicas) - aspect ratio corrigido
echo -e "\n[4/6] Gerando Elephant Valley..."
mpirun -np $NP ./bin/mandelbrot -w 8192 -h 8192 -i 4000 \
    -xmin 0.25 -xmax 0.35 -ymin -0.05 -ymax 0.05 \
    -o mandelbrot_elephant.png

# 5. Mini Mandelbrot (fractal dentro do fractal) - coordenadas corrigidas
echo -e "\n[5/6] Gerando Mini Mandelbrot..."
mpirun -np $NP ./bin/mandelbrot -w 8192 -h 8192 -i 8000 \
    -xmin -1.7687 -xmax -1.7675 -ymin -0.0006 -ymax 0.0006 \
    -o mandelbrot_mini.png

# 6. Double Spiral (espiral dupla - muito bonito)
echo -e "\n[6/6] Gerando Double Spiral..."
mpirun -np $NP ./bin/mandelbrot -w 8192 -h 8192 -i 6000 \
    -xmin -0.0905 -xmax -0.0875 -ymin 0.6535 -ymax 0.6565 \
    -o mandelbrot_double_spiral.png

echo -e "\n======================================"
echo "✅ Todas as imagens foram geradas!"
echo "======================================"
echo "Arquivos criados:"
ls -lh mandelbrot_*.png 2>/dev/null || echo "Nenhum arquivo encontrado"