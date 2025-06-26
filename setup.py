from setuptools import setup, find_packages

setup(
    name='LabSeismo',
    version='0.1.0',
    description='A package for laboratory simulated earthquakes, acoustic emissions, and Coda Wave Interferometry analysis.',
    author='Wen Zhou',
    author_email='w.zhou@uu.nl',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy'
        # Add additional dependencies if needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
