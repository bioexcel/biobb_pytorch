import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="biobb_pytorch",
    version="5.2.0",
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
    install_requires=["biobb_common==5.2.0", "torch==2.3.0", "lightning==2.5.0", "mlcolvar==1.3.0", "mdtraj"],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "MDFeaturizer = biobb_pytorch.mdae.mdfeaturizer:main",
            "buildModel = biobb_pytorch.mdae.build_model:main",
            "trainModel = biobb_pytorch.mdae.train_model:main",
            "evaluateDecoder = biobb_pytorch.mdae.decode_model:main",
            "evaluateEncoder = biobb_pytorch.mdae.encode_model:main",
            "evaluateModel = biobb_pytorch.mdae.evaluate_model:main",
            "makePlumed = biobb_pytorch.mdae.make_plumed:main",
            "feat2traj = biobb_pytorch.mdae.feat2traj:main",
            "LRP = biobb_pytorch.mdae.explainability.LRP:main",
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Operating System :: Unix",
    ],
)
