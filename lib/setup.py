from setuptools import setup, find_packages
from distutils.util import convert_path

main_namespace = {}
version_file_path = convert_path('const.py')
with open(version_file_path) as version_file:
    exec(version_file.read(), main_namespace)

setup(name='uplift_report_lib',
      version=main_namespace['__version__'],
      description='A set of helper functions for Uplift report',
      url='https://github.com/remerge/uplift-report',
      author='Remerge Tech',
      author_email='tech@remerge.io',
      license='Commercial',
      packages=find_packages(),
      include_package_data=True,
      python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*',
      install_requires=[
          'IPython<7.9.0,>=5.5.0',
          'matplotlib<3.2.0,>=3.0.3',
          'numpy<1.18.0,>=1.16.4',
          'pandas<0.25.0,>=0.24.2',
          'scipy<1.4.0,>=1.3.0',
          's3fs<0.4.0,>=0.3.0',
          'pyarrow<0.15.0,>=0.14.0',
          'partd<1.1.0,>=1.0.0',
          'xxhash<1.4.0,>=1.3.0',
      ],
      extras_require={
          'dev': [
              'flake8<3.8.0,>=3.7.8',
              'autopep8<1.5.0,>=1.4.4',
          ]
      }
      )
