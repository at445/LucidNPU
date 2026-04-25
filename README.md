# LucidNPU
A transparent, MLIR-based compiler backend demystifying NPU execution.


LucidNPU is an educational yet industrial-grade compiler backend that bridges the gap between high-level AI tensor operations and bare-metal NPU execution. It focuses on the core challenges of AI compilers: explicit SRAM memory planning (Bufferization), DMA orchestration, and operator lowering (Tiling/Fusion), making the "black box" of NPU software stacks lucid and transparent.