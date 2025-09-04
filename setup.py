#!/usr/bin/env python

from os import environ
import torch
import subprocess
import sys

# Detect CUDA version and set appropriate architecture
def get_cuda_arch():
    try:
        # Try to get CUDA version from nvcc
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout
            if 'release' in output:
                # Extract version like "release 11.7" or "release 12.8"
                version_line = [line for line in output.split('\n') if 'release' in line][0]
                version = version_line.split('release')[1].strip().split(',')[0].strip()
                major, minor = map(int, version.split('.'))
                
                # Set architecture based on CUDA version
                if major >= 12:
                    return "6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0+PTX"
                elif major >= 11:
                    return "6.0;6.1;7.0;7.5;8.0;8.6+PTX"
                else:
                    return "6.0;6.1;7.0;7.5+PTX"
    except:
        pass
    
    # Fallback to broad compatibility
    return "6.0;6.1;7.0;7.5;8.0;8.6+PTX"

# Set CUDA architecture for compatibility
environ['TORCH_CUDA_ARCH_LIST'] = get_cuda_arch()

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

long_description = open('README.rst').read()

setup(
    name='fast_jl',
    version="0.1.3",
    description="Fast JL: Compute JL projection fast on a GPU",
    author="MadryLab",
    author_email='trak@mit.edu',
    install_requires=["torch>=2.0.0"],
    long_description=long_description,
    ext_modules=[
        CUDAExtension('fast_jl', [
            'fast_jl.cu',
        ], extra_compile_args={
            'cxx': ['-O3', '-std=c++14'],
            'nvcc': ['-O3', '--use_fast_math', '-std=c++14', '--expt-relaxed-constexpr']
        }),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    setup_requires=["torch>=2.0.0"])
