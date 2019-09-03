from setuptools import setup
from fifty.__init__ import __version__ as VERSION

setup(
    name='fifty',
    version=VERSION,
    description='FiFTy: Large-scale File Fragment Type Identification using Neural Networks',
    url='https://github.com/mittalgovind/fifty',
    author='Govind Mittal, Pawel Korus and Nasir Memon',
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
        'docopt',
    ],
    packages=['fifty', 'fifty.utilities', 'fifty.commands', 'fifty.utilities.models'],
    zip_safe=False,
    entry_points={
        'console_scripts': ['fifty=fifty.cli:main'],
    },
    include_package_data=True,
)
