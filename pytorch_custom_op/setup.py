# SPDX-License-Identifier: Apache-2.0

from setuptools import setup
from torch.utils import cpp_extension

setup(name="custom_group_norm",
      ext_modules=[cpp_extension.CppExtension("custom_group_norm", ["custom_group_norm.cpp"], 
                                              include_dirs=["/usr/local/include/eigen3"])],
      license="Apache License v2.0",
      cmdclass={"build_ext": cpp_extension.BuildExtension})
