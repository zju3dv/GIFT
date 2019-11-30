import os

cuda_version='8.0' # installed cuda version
cuda_include=f'/usr/local/cuda-{cuda_version}/include'
cuda_library=f'/usr/local/cuda-{cuda_version}/lib64'

os.system('nvcc src/nearest_neighborhood.cu -c -o src/nearest_neighborhood.cu.o -x cu -Xcompiler -fPIC -O2 -arch=sm_52 -I {} -D_FORCE_INLINES'.
          format(cuda_include))

from cffi import FFI
ffibuilder = FFI()


# cdef() expects a string listing the C types, functions and
# globals needed from Python. The string follows the C syntax.
with open(os.path.join(os.path.dirname(__file__), "src/utils_python_binding.h")) as f:
    ffibuilder.cdef(f.read())

ffibuilder.set_source("_extend_utils",
                      """
                             #include "src/utils_python_binding.h"   // the C header of the library
                      """,
                      extra_objects=['src/nearest_neighborhood.cu.o'],
                      libraries=['stdc++','cudart'],
                      extra_link_args=[f"-L{cuda_library}"],
                      # extra_compile_args=[],
                      )

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
    os.system("rm src/*.o")
