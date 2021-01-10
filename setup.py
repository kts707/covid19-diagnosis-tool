from setuptools import setup

setup(
   name='Covid-19 Detector',
   version='0.1.0',
   description='The covid-19 lung ct scan classification and segmentation models',
   author='Chris',
   install_requires=[
    'numpy', 
    'torch',
    'torchvision',
    'matplotlib',
    'sklearn',
    'IPython',
    ], #external packages as dependencies
)
