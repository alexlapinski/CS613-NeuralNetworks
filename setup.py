# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

setup(
    name='cs613_hw4',
    version='1.0.0',
    description='CS613 - HW4 - Artificial Neural Networks',
    long_description=readme,
    author='Alex Lapinski',
    author_email='contact@alexlapinski.name',
    url='https://github.com/alexlapinski/CS613-NeuralNetworks',
    license=None,
    packages=find_packages(exclude=('tests', 'data', 'docs'))
)
