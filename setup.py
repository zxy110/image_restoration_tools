#!/usr/bin/env python
from __future__ import print_function
from setuptools import setup, find_packages
import sys
 
setup(
    name="image_restoration_tools",
    version="0.1.1",
    author="Leo",
    author_email="zxy2019@pku.edu.cn",
    description="a package of image restoration",
    long_description=open("README.rst").read(),
    license="MIT",
    url="https://github.com/zxy110/image_restoration_tools",
    packages=['image_restoration_tools'],
)