from setuptools import setup
from setuptools import find_packages

setup(
   name='terl',
   version='0.1.2',
   author='Ariel Boukris',
   author_email='ariel.boukris@outlook.com',
   description='T.E.R.L - Trading Env for Renforcement Learning',
   long_description=open('README.md').read(),
   packages=find_packages(),
   python_requires='>=3.6',
   install_requires=[
       "gym",
       "vaex",
       "pandas",
       "numpy",
       "numba",
       'ta',
   ]
)