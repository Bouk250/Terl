from setuptools import setup

setup(
   name='terl',
   version='0.1.0',
   author='Ariel Boukris',
   author_email='ariel.boukris@outlook.com',
   description='T.E.R.L - Trading Env for Renforcement Learning',
   long_description=open('README.md').read(),
   install_requires=[
       "gym",
       "vaex",
       "pandas",
       "numpy"
   ]
)