#!/usr/bin/env python
# Created by "Thieu" at 13:24, 27/02/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        README = f.read()
    return README


setup(
    name="opfunu",
    version="1.0.2",
    author="Thieu",
    author_email="nguyenthieu2102@gmail.com",
    description="Opfunu: An Open-source Python Library for Optimization Benchmark Functions",
    long_description=readme(),
    long_description_content_type="text/markdown",
    keywords=["optimization functions", "test functions", "benchmark functions", "mathematical functions",
              "CEC competitions", "CEC-2008", "CEC-2009", "CEC-2010", "CEC-2011", "CEC-2012", "CEC-2013",
              "CEC-2014", "CEC-2015", "CEC-2017", "CEC-2019", "CEC-2020", "CEC-2021", "CEC-2022", "soft computing",
              "Stochastic optimization", "Global optimization", "Convergence analysis", "Search space exploration",
              "Local search", "Computational intelligence", "Performance analysis",
              "Exploration versus exploitation", "Constrained optimization", "Simulations"],
    url="https://github.com/thieu1995/opfunu",
    project_urls={
            'Documentation': 'https://opfunu.readthedocs.io/',
            'Source Code': 'https://github.com/thieu1995/opfunu',
            'Bug Tracker': 'https://github.com/thieu1995/opfunu/issues',
            'Change Log': 'https://github.com/thieu1995/opfunu/blob/master/ChangeLog.md',
            'Forum': 'https://t.me/+fRVCJGuGJg1mNDg1',
    },
    packages=find_packages(exclude=['tests*', 'examples*']),
    include_package_data=True,
    license="GPLv3",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: System :: Benchmark",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    install_requires=["numpy>=1.16.5", "matplotlib>=3.3.0", "Pillow>=9.1.0", "requests>=2.27.0"],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.1"],
    },
    python_requires='>=3.7, <3.12',
)
