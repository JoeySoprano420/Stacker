# -----------------------------
# Stacker Compiler Connector
# -----------------------------
import os, sys, subprocess
from llvmlite import binding

# Assume: tokenize(), parse(), compile_program() are already defined
# compile_program(ast) → returns LLVM IR and writes stacker.out.o

RUNTIME_C = "stacker_runtime.c"
RUNTIME_OBJ = "stacker_runtime.o"

def build_runtime():
    """
    Compile stacker_runtime.c into an object file.
    This is done once and reused.
    """
    if not os.path.exists(RUNTIME_OBJ):
        print("🔧 Building Stacker runtime...")
        subprocess.run(["clang", "-c", RUNTIME_C, "-o", RUNTIME_OBJ, "-O2"], check=True)

def link_executable(obj_file, output="stacker.out"):
    """
    Link the compiled object file with runtime into final executable.
    """
    build_runtime()
    subprocess.run(["clang", obj_file, RUNTIME_OBJ, "-o", output, "-O2"], check=True)
    print(f"✅ Built executable: {output}")

def compile_and_run(filename):
    """
    Full pipeline: .stk → LLVM IR → .o → native exe
    """
    with open(filename, "r") as f:
        src = f.read()

    # Lex & Parse
    tokens = tokenize(src)
    ast = parse(tokens)

    # Compile to LLVM IR + Object
    llvm_ir = compile_program(ast)  # writes stacker.out.o

    # Link against runtime
    link_executable("stacker.out.o", "stacker.out")

    # Run the result
    subprocess.run(["./stacker.out"])
