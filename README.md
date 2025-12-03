# Mandelbrot Set - MPI + CUDA

Implementação paralela do conjunto de Mandelbrot usando MPI (memória distribuída) e CUDA (GPU).

O Conjunto de Mandelbrot é um dos fractais mais conhecidos e estudados na matemática. Ele é gerado a partir de uma fórmula simples:

  zₙ₊₁ = zₙ² + c

Nessa equação, c é um número complexo fixo e a iteração começa com z₀ = 0. Para cada valor de c, executamos essa iteração várias vezes e observamos o comportamento da sequência.
Se ela permanece limitada (ou seja, não cresce sem controle), então c faz parte do Conjunto de Mandelbrot. Caso contrário, ele fica fora.

Ao representar graficamente todos esses valores, surge uma figura fractal: uma forma que possui detalhes infinitos e padrões que se repetem conforme aproximamos a imagem. Apesar de sua origem simples, o Mandelbrot se tornou um símbolo da complexidade gerada por regras básicas.

Esse conjunto é amplamente usado para estudar sistemas dinâmicos, visualizar fractais e testar algoritmos de computação de alto desempenho, já que seu cálculo envolve muitas operações repetidas e independentes.

## 📋 Descrição

Este projeto calcula e renderiza o conjunto de Mandelbrot utilizando duas formas de paralelismo:

- **MPI**: Distribui linhas da imagem entre diferentes processos
- **CUDA**: Processa pixels em paralelo na GPU usando threads CUDA

### Características Técnicas

- **Smooth Coloring**: Algoritmo de colorização suave que elimina bandas de cor
- **8 Paletas de Cores**: Ultra, Fire, Ice, Psychedelic, Rainbow, Monochrome, Ocean e Sunset
- **Otimizações**: Detecção de cardioide e bulbo período-2 para escape rápido
- **Detecção de Periodicidade**: Floyd's cycle detection para pontos dentro do conjunto
- **Saída PNG**: Formato de imagem portável com compressão

## 🛠️ Requisitos

- CUDA Toolkit (11.0+)
- OpenMPI ou MPICH
- GCC/G++
- Make
- GPU NVIDIA

## 📁 Estrutura do Projeto

```
.
├── main.c                      # Programa principal com MPI
├── Makefile                    # Script de compilação
├── README.md                   # Este arquivo
├── bin/
│   └── mandelbrot              # Executável compilado
├── include/
│   ├── mandelbrot_kernel.h     # Header do kernel CUDA
│   ├── image_utils.h           # Header das utilidades de imagem
│   ├── metrics.h               # Header das métricas
│   └── stb_image_write.h       # Biblioteca para escrita de PNG
├── obj/                        # Arquivos objeto compilados
├── scripts/
│   └── test_zooms.sh           # Script para testar várias regiões
├── src/
│   └── mandelbrot_kernel.cu    # Kernel CUDA para cálculo do Mandelbrot
└── utils/
    ├── image_utils.c           # Funções para salvar imagem PNG
    └── metrics.c               # Sistema de métricas de desempenho
```

## 🚀 Compilação

```bash
make
```

Isso criará o executável em `bin/mandelbrot`.

## ▶️ Execução

### Execução básica:

```bash
# Com parâmetros padrão
mpirun -np 4 ./bin/mandelbrot

# Ver ajuda
./bin/mandelbrot --help
```

### Com parâmetros personalizados:

```bash
# Mudar resolução e iterações
mpirun -np 4 ./bin/mandelbrot -w 10000 -h 10000 -i 3000

# Fazer zoom em região específica (espiral colorida)
mpirun -np 4 ./bin/mandelbrot -xmin -0.8 -xmax -0.4 -ymin -0.2 -ymax 0.2 -i 3000 -o zoom_espiral.png

# Seahorse Valley (muito detalhado e colorido)
mpirun -np 4 ./bin/mandelbrot -xmin -0.75 -xmax -0.73 -ymin 0.1 -ymax 0.12 -i 5000 -o seahorse.png

# Elephant Valley
mpirun -np 4 ./bin/mandelbrot -xmin 0.28 -xmax 0.30 -ymin 0.008 -ymax 0.012 -i 4000 -o elephant.png
```

### Atalhos do Makefile:

```bash
make run1   # 1 processo
make run2   # 2 processos
make run4   # 4 processos
make run8   # 8 processos
make test   # Testa 1, 2 e 4 processos
```

### Script para testar várias regiões:

```bash
chmod +x scripts/test_zooms.sh
./scripts/test_zooms.sh
```

Isso vai gerar 6 imagens diferentes automaticamente!

## ⚙️ Configuração

### Parâmetros disponíveis:

```
-w <width>      Largura da imagem (padrão: 12288)
-h <height>     Altura da imagem (padrão: 12288)
-i <iter>       Iterações máximas (padrão: 2000)
-xmin <valor>   Limite esquerdo (padrão: -2.5)
-xmax <valor>   Limite direito (padrão: 1.0)
-ymin <valor>   Limite inferior (padrão: -1.25)
-ymax <valor>   Limite superior (padrão: 1.25)
-o <arquivo>    Nome do arquivo de saída (padrão: mandelbrot.png)
-p <paleta>     Paleta de cores (padrão: ultra)
--help          Mostra mensagem de ajuda
```

### 🎨 Paletas de Cores Disponíveis:

| Paleta        | Descrição                       |
| ------------- | ------------------------------- |
| `ultra`       | Azul/laranja/branco (clássica)  |
| `fire`        | Vermelho/amarelo/laranja (fogo) |
| `ice`         | Azul/ciano/branco (gelo)        |
| `psychedelic` | Cores vibrantes e saturadas     |
| `rainbow`     | Arco-íris completo              |
| `monochrome`  | Preto e branco                  |
| `ocean`       | Tons de azul e verde            |
| `sunset`      | Cores de pôr do sol             |

### Exemplos com paletas:

```bash
# Paleta de fogo
mpirun -np 4 ./bin/mandelbrot -p fire -o mandelbrot_fire.png

# Paleta psicodélica com zoom
mpirun -np 4 ./bin/mandelbrot -p psychedelic -xmin -0.8 -xmax -0.4 -ymin -0.2 -ymax 0.2 -i 3000 -o psychedelic.png
```

### Regiões interessantes para explorar:

**🌀 Espiral Colorida:**

```bash
mpirun -np 4 ./bin/mandelbrot -xmin -0.8 -xmax -0.4 -ymin -0.2 -ymax 0.2 -i 3000
```

**🐴 Seahorse Valley (vale dos cavalos-marinhos):**

```bash
mpirun -np 4 ./bin/mandelbrot -xmin -0.75 -xmax -0.73 -ymin 0.1 -ymax 0.12 -i 5000
```

**🐘 Elephant Valley (vale dos elefantes):**

```bash
mpirun -np 4 ./bin/mandelbrot -xmin 0.28 -xmax 0.30 -ymin 0.008 -ymax 0.012 -i 4000
```

**🔬 Mini Mandelbrot (fractal dentro do fractal):**

```bash
mpirun -np 4 ./bin/mandelbrot -xmin -0.1592 -xmax -0.1568 -ymin 1.0317 -ymax 1.0341 -i 6000
```

**Dica**: Quanto maior o zoom (menor a diferença entre min e max), mais iterações você precisa para ver detalhes!

## 📊 Métricas Exibidas

O programa exibe métricas detalhadas após a execução:

- **Configuração**: Número de processos, dimensões, iterações, paleta selecionada, região do plano complexo
- **Tempos de Execução**: Total, computação GPU, comunicação MPI, I/O (salvar PNG)
- **Desempenho**: Pixels/segundo (Mpixels/s), GFlops estimado
- **Eficiência**: Eficiência paralela, speedup de comunicação
- **Memória**: Uso de memória total e por processo

## 🖼️ Saída

O programa gera um arquivo PNG colorido (RGB, 3 bytes por pixel) que pode ser visualizado em qualquer visualizador de imagens. O formato padrão de saída é `mandelbrot.png`.

## 🧹 Limpeza

```bash
make clean
```

Remove todos os arquivos compilados e a imagem gerada.

## 📝 Como Funciona

1. **MPI divide o trabalho**: Cada processo MPI recebe um conjunto de linhas da imagem para processar
2. **Broadcast de parâmetros**: O processo 0 distribui as configurações para todos via `MPI_Bcast`
3. **Alocação GPU**: Cada processo aloca memória na GPU para sua porção da imagem
4. **CUDA processa**: O kernel CUDA calcula o Mandelbrot em paralelo usando blocos de 32x8 threads
5. **Smooth Coloring**: Utiliza algoritmo de colorização suave para gradientes sem bandas
6. **Coleta de resultados**: O processo 0 coleta todas as partes via `MPI_Gatherv`
7. **Salvamento PNG**: A imagem completa é salva usando a biblioteca stb_image_write
8. **Métricas**: Exibe estatísticas detalhadas de desempenho

### Algoritmo de Colorização

O projeto utiliza **Smooth Iteration Count** para eliminar bandas de cor:

```
smooth_iter = iter + 1 - log(log(|z|) / log(2)) / log(2)
```

Isso produz valores fracionários de iteração que permitem transições suaves de cores.

## 🎓 Trabalho Acadêmico

Este projeto foi desenvolvido para demonstrar o uso combinado de:

- Paralelismo de memória distribuída (MPI)
- Paralelismo em GPU (CUDA)
- Algoritmos de colorização fractal (Smooth Coloring)
- Análise de desempenho e métricas

### Conceitos Demonstrados

- **MPI_Bcast**: Distribuição de parâmetros para todos os processos
- **MPI_Gatherv**: Coleta de dados de tamanhos variáveis
- **CUDA Kernels**: Paralelismo massivo em GPU
- **Divisão de trabalho**: Balanceamento de carga entre processos
- **Otimizações matemáticas**: Detecção de cardioide e periodicidade

## 🔧 Troubleshooting

### Overhead alto (>70%):

O overhead alto é normal quando você tem:

- Imagem pequena com poucos processos
- GPU muito rápida

**Para reduzir o overhead:**

1. Aumente resolução: `-w 16384 -h 16384`
2. Aumente iterações: `-i 5000`
3. Faça zoom em regiões complexas (seahorse valley precisa de mais computação)
4. Use menos processos MPI (a GPU já é paralela!)

**Nota:** Com 1 processo, você terá menos overhead de comunicação MPI.

### Erro de compute capability:

Se você tiver uma GPU diferente, ajuste a flag `-arch` no Makefile:

```makefile
NVCC_FLAGS = -arch=sm_XX  # Substitua XX pela sua compute capability
```

### MPI não encontrado:

Instale OpenMPI:

```bash
sudo apt-get install openmpi-bin libopenmpi-dev
```

### CUDA não encontrado:

Certifique-se de que o CUDA Toolkit está instalado e `/usr/local/cuda/bin` está no PATH.

## 👤 Autores

- Cristian dos Santos Siquiera — https://github.com/CristianSSiqueira
- Pedro Rockenbach Frosi — https://github.com/frosipedro

