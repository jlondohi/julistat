from setuptools import setup, find_packages

setup(
    name="julistat",
    version="0.1.0",
    description="""Julistat is a customized package that offers useful tools for data description and processing. 
                   It is designed to perform various one-dimensional or low-dimensional analyses,
                   allowing the determination of importance, relevance, segmentations or other 
                   necessary groupings as first steps in building models.""",
    author="Julian D. Londono",
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3.12.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)