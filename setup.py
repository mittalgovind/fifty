from setuptools import setup

setup(
    name='fifty',
    version=__version__,
    description='FiFTy: Large-scale File Fragment Type Identification using Neural Networks',
    url='https://github.com/mittalgovind/fifty',
    author='Govind Mittal & Pawel Korus',
    author_email='mittal@nyu.edu',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Utilities',
        'License :: Public Domain',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
    install_requires=[
        'pandas',
        'numpy',
        'tensorflow',
        'keras',
        'pathlib',
        'hyperopt',
    ],
    packages=['fifty'],
    zip_safe=False,
    entry_points={
        'console_scripts': ['fifty=fifty:main'],
    }
)
