from setuptools import setup, find_packages

setup(name='gym_sokoban',
      version='0.1.0',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          'gym>=0.2.3', 'scipy>=1.0.0', 'numpy>=1.14.1', 'numba', 'matplotlib',
          'attr', 'munch', 'pillow', 'gin-config'
      ],
      extras_require={
          'dev': ['pytest'],
      }
)
