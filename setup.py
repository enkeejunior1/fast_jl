#!/usr/bin/env python

from os import environ
import torch
import subprocess
import sys
import os

# Force CUDA version compatibility by setting environment variables
def force_cuda_compatibility():
    # Set CUDA_HOME to match PyTorch's CUDA version
    torch_cuda_version = torch.version.cuda
    print(f"PyTorch CUDA version: {torch_cuda_version}")
    
    # Try to find CUDA installation that matches PyTorch
    possible_cuda_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-12.8",
        "/usr/local/cuda-12.0",
        "/usr/local/cuda-11.8",
        "/usr/local/cuda-11.7",
        "/opt/cuda",
        "/opt/cuda-12.8",
        "/opt/cuda-12.0",
        "/opt/cuda-11.8",
        "/opt/cuda-11.7",
    ]
    
    # Also check conda environment
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        possible_cuda_paths.extend([
            f"{conda_prefix}",
            f"{conda_prefix}/pkgs/cuda-toolkit",
        ])
    
    # Set CUDA_HOME to the first available path
    for path in possible_cuda_paths:
        if os.path.exists(path):
            environ['CUDA_HOME'] = path
            print(f"Setting CUDA_HOME to: {path}")
            break
    
    # Force CUDA version to match PyTorch
    environ['CUDA_VERSION'] = torch_cuda_version
    environ['TORCH_CUDA_ARCH_LIST'] = "6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    
    # Disable CUDA version check
    environ['TORCH_CUDA_VERSION_CHECK'] = '0'

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

# Force CUDA compatibility
force_cuda_compatibility()

# Set CUDA architecture for compatibility
environ['TORCH_CUDA_ARCH_LIST'] = get_cuda_arch()

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Custom build extension that bypasses CUDA version check
class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        # Monkey patch the CUDA version check
        import torch.utils.cpp_extension
        original_check = torch.utils.cpp_extension._check_cuda_version
        
        def dummy_check(*args, **kwargs):
            print("Skipping CUDA version check...")
            return
        
        torch.utils.cpp_extension._check_cuda_version = dummy_check
        
        # Call the original build_extensions
        super().build_extensions()
        
        # Restore the original function
        torch.utils.cpp_extension._check_cuda_version = original_check

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
            'nvcc': ['-O3', '--use_fast_math', '-std=c++14', '--expt-relaxed-constexpr', '--allow-unsupported-compiler']
        }),
    ],
    cmdclass={
        'build_ext': CustomBuildExtension
    },
    setup_requires=["torch>=2.0.0"])
