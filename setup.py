from setuptools import find_packages, setup
setup(
    name='abtools',
    packages=find_packages(include=['abtools']),
    version='0.1.0',
    description='Base methods for ab-analysis',
    author='n-surkov',
    license='',
    install_requires=[],
    setup_requires=['matplotlib', 'seaborn', 'numpy', 'scipy', 'pandas']
)
