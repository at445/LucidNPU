# GDB init for Cursor/VS Code cppdbg (loaded via miDebuggerArgs -ix).
# Loads LLVM + MLIR pretty printers so Type/Value/Attribute/SmallVector etc.
# expand in the Variables panel instead of showing opaque impl pointers.

set print pretty on

python
import os
import gdb

home = os.environ.get("HOME", "")
llvm_pp = os.path.join(home, "llvm-project/llvm/utils/gdb-scripts/prettyprinters.py")
mlir_pp = os.path.join(home, "llvm-project/mlir/utils/gdb-scripts/prettyprinters.py")

for label, path in [("LLVM", llvm_pp), ("MLIR", mlir_pp)]:
    if not os.path.isfile(path):
        gdb.write(f"warning: {label} pretty printer not found: {path}\n")
        continue
    gdb.execute(f"source {path}")

if home:
    gdb.execute(f"skip -gfile {home}/llvm-project/*")
gdb.execute("skip -gfile /usr/include/c++/*")
end
