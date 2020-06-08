from __future__ import print_function
from distutils.core import setup
from distutils.command.install import install as DistutilsInstall
import sys
import subprocess
import tensorflow
    

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
