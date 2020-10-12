from setuptools import setup

setup(name='stardate2',
      version='0.1',
      description='Inferring stellar ages with gyrochronology',
      url='http://github.com/RuthAngus/stardate2',
      author='Ruth Angus',
      author_email='ruthangus@gmail.com',
      license='MIT',
      packages=['stardate2'],
      install_requires=['numpy', 'pandas', 'tqdm'],
      zip_safe=False)
