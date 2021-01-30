from setuptools import setup, find_packages
import versioneer


setup(
    name='DNet',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Deep Neural Network implementation in numpy',
    author='Javier Cárdenas',
    install_requires=[
        "numpy"
    ],
)