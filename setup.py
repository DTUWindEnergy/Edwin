# -*- coding: utf-8 -*-
"""
Setup file for EDWIN
"""
import os
from setuptools import setup, find_packages
import pkg_resources

repo = os.path.dirname(__file__)
try:
    from git_utils import write_vers
    version = write_vers(vers_file='ed_win/__init__.py', repo=repo, skip_chars=1)
except Exception:
    version = '999'


try:
    from pypandoc import convert_file

    def read_md(f): return convert_file(f, 'rst', format='md')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")

    def read_md(f): return open(f, 'r').read()


setup(name='ed_win',
      version=version,
      description='EDWIN an optimization and design package for electrical networks in windfarms',
      long_description=read_md('README.md'),
      url='https://gitlab.windenergy.dtu.dk/TOPFARM/EDWIN',
      author='DTU Wind Energy',
      author_email='juru@dtu.dk',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'matplotlib',  # for plotting
          'numpy',  # for numerical calculations
          'xarray',  # for WaspGridSite data storage
          'scipy',  # constraints
      ],
      zip_safe=True)
