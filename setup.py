import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="biobb_pytorch",
    version="4.1.0",
    author="Biobb developers",
    author_email="pau.andrio@bsc.es",
    description="biobb_pytorch is the Biobb module collection to create and train ML & DL models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="Bioinformatics Workflows BioExcel Compatibility",
    url="https://github.com/bioexcel/biobb_pytorch",
    project_urls={
        "Documentation": "http://biobb_pytorch.readthedocs.io/en/latest/",
        "Bioexcel": "https://bioexcel.eu/"
    },
    packages=setuptools.find_packages(exclude=['docs', 'test']),
    install_requires=['biobb_common==4.1.0'],
    python_requires='>=3.8',
    entry_points={
        "console_scripts": [
            "NAME_OF_COMMAND = biobb_pytorch.NAME_OF_PACKAGE.NAME_OF_MODULE:main",
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Operating System :: Unix"
    ],
)
