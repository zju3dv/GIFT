from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='hard_example_mining',
    ext_modules=[
        CUDAExtension('hard_example_mining', [
            './src/hard_example_mining.cpp',
            './src/hard_example_mining_kernel.cu',
            './src/knncuda.cu'
        ], libraries=['cublas',])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
