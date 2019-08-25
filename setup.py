from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='techMiner',
      version='0.0.0',
      description='Tech mining of bibliograpy',
      long_description='Tech mining of bibliograpy',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
      ],
      keywords='bibliograpy',
      url='http://github.com/jdvelasq/techMiner',
      author='Ivanohe J. Garces & Juan D. Velasquez',
      author_email='jdvelasq@unal.edu.co',
      license='MIT',
      packages=['techMiner'],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
