from setuptools import setup, find_packages

setup(
    name='floodfill-skeleton',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='Neuron skeleton-segmentation interface. Used for constrained segmentation and error detection',
    long_description=open('README.md').read(),
    install_requires=['numpy','scipy'],
    url='https://github.com/pattonw/Floodfill-Skeleton',
    author='William Patton',
    author_email='wllmpttn24@gmail.com'
)
