from setuptools import setup


setup(name='MalenoV',
      version='0.0',
      description='Function for seismic facies training /classification using Convolutional Neural Nets (CNN)',
      author='Charles Rutherford Ildstad',
      install_requires=['numpy',
                        'scipy',
                        'matplotlib',
                        'tensorflow',
                        'keras',
                        'segyio==1.3.0'],
      extras_require={
          'gpu': ['tensorflow-gpu'],
      },
      entry_points={
          'console_scripts': [
              'malenov = malenov.__main__:main',
          ]
      },
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],
      packages=['malenov'])
