#!/usr/bin/env python

import os
import sys
from torch.utils.cpp_extension import BuildExtension as TorchBuildExtension

class CustomBuildExtension(TorchBuildExtension):
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
