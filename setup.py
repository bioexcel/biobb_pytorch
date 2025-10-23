import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="biobb_pytorch",
    version="5.1.1",
    author="Biobb developers",
    author_email="pieter.zanders@bsc.es",
    description="biobb_pytorch is the Biobb module collection to create and train ML & DL models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="Bioinformatics Workflows BioExcel Compatibility",
    url="https://github.com/bioexcel/biobb_pytorch",
    project_urls={
        "Documentation": "http://biobb-autoencoders.readthedocs.io/en/latest/",
        "Bioexcel": "https://bioexcel.eu/",
    },
    packages=setuptools.find_packages(exclude=["docs", "test"]),
    package_data={"biobb_pytorch": ["py.typed"]},
    install_requires=["biobb_common==5.1.0", "torch", "lightning", "mlcolvar"],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "mdfeaturizer = biobb_pytorch.mdae.mdfeaturizer:main",
            "build_model = biobb_pytorch.mdae.build_model:main",
            "train_model = biobb_pytorch.mdae.train_model:main",
            "evaluate_model = biobb_pytorch.mdae.evaluate_model:main",
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Operating System :: Unix",
    ],
)
