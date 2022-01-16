from setuptools import setup, find_packages

long_description = '''
Useful and frequently used pytorch code for training neural networks.
'''

setup(name='nnlib',
      version='0.1',
      description='Useful and frequently used pytorch code for training neural networks',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Hrayr Harutyunyan',
      author_email='harhro@gmail.com',
      url='https://github.com/hrayrhar/nnlib',
      license='GNU Affero General Public License v3.0',
      install_requires=[],
      tests_require=[],
      classifiers=[
      ],
      packages=find_packages(),
      )
