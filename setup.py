#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()


def parse_requirements(filename):
    lines = (line.strip() for line in open(filename))
    return [line for line in lines if line and not line.startswith("#")]


requirements = parse_requirements("requirements/dev.txt")

setup_requirements = parse_requirements("requirements/dev.txt")

test_requirements = parse_requirements("requirements/dev.txt")

setup(
    author="William Hunter Patton",
    author_email="pattonw@hhmi.org",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="A package for combining a 3D arbor with a dense volumetric segmentation.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="sarbor",
    name="sarbor",
    packages=find_packages(include=["sarbor"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/pattonw/sarbor",
    version="0.1.0",
    zip_safe=False,
    entry_points="""
        [console_scripts]
        sarbor-error-detector=sarbor.cli:cli
    """,
)
