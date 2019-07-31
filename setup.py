from setuptools import setup

setup(name='uplift_report_lib',
      version='1.0.0',
      description='A set of helper functions for Uplift report',
      url='https://github.com/remerge/uplift-report',
      author='Remerge Tech',
      author_email='tech@remerge.io',
      license='Commercial',
      python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*',
      install_requires=[
          'xxhash==1.3.0',
          'pandas==0.24.0',
          'scipy==1.2.2',
          's3fs==0.2.2',
          'pyarrow==0.14.1',
          'partd==1.0.0',
      ],
      extras_require={
          'dev': [
              'flake8==3.7.8',
              'autopep8==1.4.4',
          ]
      }
      )