set(LLVM_TARGET_DEFINITIONS DialectAndOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS CustomizedCanonicalize.td)
mlir_tablegen(CustomizedCanonicalize.inc -gen-rewriters)

set(LLVM_TARGET_DEFINITIONS ShapeInfrencceOpInterface.td)
mlir_tablegen(ShapeInfrencceOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInfrencceOpInterface.cpp.inc -gen-op-interface-defs)

add_public_tablegen_target(ToyDialectIncGen)

