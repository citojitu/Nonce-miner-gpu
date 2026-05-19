# Build: make
# Required: CUDA toolkit (nvcc) — install with `apt install nvidia-cuda-toolkit`
# Tested target: RTX 30/40 series (compute_75 to compute_90).
#
# Output: ./nonce-miner-gpu  (drop-in for the Node wrapper at ../miner.js)

NVCC      ?= nvcc
SM        ?= 75        # compute capability: 75 (Turing), 86 (Ampere), 89 (Ada), 90 (Hopper)
OPT       ?= -O3 -use_fast_math
ARCH      = -gencode arch=compute_$(SM),code=sm_$(SM)
CFLAGS    = $(OPT) $(ARCH) --std=c++17 -Xcompiler "-O3 -march=native"

TARGET    = nonce-miner-gpu
SRC       = main.cu keccak.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $@ $(SRC)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
