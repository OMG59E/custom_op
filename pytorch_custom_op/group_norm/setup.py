# SPDX-License-Identifier: Apache-2.0
import os
from setuptools import setup
from torch.utils import cpp_extension

filepath = os.path.dirname(os.path.abspath(__file__))
eigen3_inc = os.path.join(filepath, "../3rdparty/eigen3")

setup(name="custom_group_norm",
      ext_modules=[cpp_extension.CppExtension("custom_group_norm", ["custom_group_norm.cpp"], 
                                              include_dirs=[eigen3_inc])],
      license="Apache License v2.0",
      cmdclass={"build_ext": cpp_extension.BuildExtension})
