from setuptools import setup, find_packages

setup(name='dial-mpc-genesis',
      author="Davide De Blasio",
      packages=find_packages(include="dial_mpc_genesis"),
      version='0.0.1',
      install_requires=[
          'numpy<2.0.0',
          'matplotlib',
          'tqdm',
          'tyro',
          'jax',
          'jax-cosmo',
          'mujoco',
          'art',
          'emoji',
          'scienceplots',
          'torch',
          'torchvision',
          'torchaudio',
          'flax'
      ],
      package_data={'dial-mpc-genesis': ['models/', 'examples/']},
      entry_points={
          'console_scripts': [
              'dial-mpc-genesis=dial_mpc_genesis.core.dial_core:main',
          ],
      },
      )
