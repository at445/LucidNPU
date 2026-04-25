# LucidNPU

A transparent, MLIR-based compiler backend demystifying NPU execution.

LucidNPU is an educational yet industrial-grade compiler backend that bridges the gap between high-level AI tensor operations and bare-metal NPU execution. It focuses on the core challenges of AI compilers: explicit SRAM memory planning (Bufferization), DMA orchestration, and operator lowering (Tiling/Fusion), making the "black box" of NPU software stacks lucid and transparent.

## 1. Compile LLVM/MLIR Dependencies

Compile the MLIR project first. This project is developed using the `llvmorg-19.1.0` version. We only need to build MLIR and Clang. Follow these recommended steps:

```bash
cd [llvm_project_path]
git checkout llvmorg-19.1.0
cmake -S llvm -B build -G Ninja \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_USE_SPLIT_DWARF=ON \
  -DLLVM_PARALLEL_LINK_JOBS=2 \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_CCACHE_BUILD=ON

cd build
ninja -j4 mlir-opt mlir-translate FileCheck
```

After compiling, you can verify your build artifact with:

```bash
./bin/mlir-opt --version
```

## 2. Compile LucidNPU

Now you can build the LucidNPU project itself:

```bash
cd [LucidNPU_project_path]
mkdir build && cd build

# Configure the project, pointing to the MLIR and LLVM CMake directories
cmake -G Ninja \
  -DMLIR_DIR=[llvm_project_path]/build/lib/cmake/mlir \
  -DLLVM_DIR=[llvm_project_path]/build/lib/cmake/llvm \
  ..

# Build the project
cmake --build .
```