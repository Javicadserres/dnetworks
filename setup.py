import versioneer

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dnetworks',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Deep Neural Network implementation in numpy',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Javier Cárdenas',
    package_dir={'':'src'},
    packages=find_packages(where='src'),
    install_requires=[
        "numpy"
    ],
)