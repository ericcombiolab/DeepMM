from setuptools import setup, find_packages

setup(
    name='DeepMM',
    version='0.1.0',
    description='DeepMM: Identify and correct Metagenome Misassemblies with deep learning',
    author='Yi Ding',
    author_email='dylan1021@comp.hkbu.edu.hk',
    url='https://github.com/ericcombiolab/DeepMM',
    packages=find_packages(),
    install_requires=[
        'tqdm',
        'torch',
        'numpy',
        'scipy',
        'pysam',
        'pandas',
        'biopython',
        'matplotlib',
        'scikit-learn', 
        'torchvision',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License', 
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': ['DeepMM=DeepMM.DeepMM:main'],
    }
)
