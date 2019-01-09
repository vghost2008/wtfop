#!/usr/bin/env python

from __future__ import print_function
from distutils.core import setup
from distutils.command.install import install as DistutilsInstall
import sys
import subprocess

try:
    import tensorflow
except ImportError:
    print("Please install tensorflow 0.12.0 or later")
    sys.exit()
    

class MyInstall(DistutilsInstall):
    def run(self):
        subprocess.call(['make', '-C', 'wtfop', 'build'])
        DistutilsInstall.run(self)

setup(name='wtfop',
            version='1.0',
            description='WTFOP as a set of custom TensorFlow operation',
            author='wangjie',
            packages=['wtfop'],
            package_data={'wtfop': ['wtfop.so']},
            cmdclass={'install': MyInstall}
)
