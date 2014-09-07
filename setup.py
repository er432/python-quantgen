#!/usr/bin/env python
from setuptools import setup, find_packages
import os
import sys


packages = ['quantgen',
            'quantgen.analysis',
            'quantgen.core',
            'quantgen.sim',
            'quantgen.stats_utils',
            'quantgen.data',
            'quantgen.test',
            'quantgen.test.core',
            'quantgen.test.stats_utils'
           ]
setup(
    name='quantgen',
    version='0.3',
    author='Eli Rodgers-Melnick',
    author_email='er432@cornell.edu',
    description='Quantitative genetics in Python',
    url='https://github.com/er432/python-quantgen',
    platforms=['Linux','Mac OSX', 'Windows', 'Unix'],
    keywords=['Genomics','Quantitative genetics'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ],
    packages=packages,
    license='BSD'
    )
