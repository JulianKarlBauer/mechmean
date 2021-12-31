import setuptools

setuptools.setup(
    name="mechmean",
    version="0.0.1",
    author="Julian Karl Bauer",
    author_email="juliankarlbauer@gmx.de",
    description="mechmean",
    url="https://github.com/JulianKarlBauer/mechmean/",
    packages=["mechmean"],
    package_dir={"mechmean": "mechmean"},
    install_requires=[
        "numpy",
        "scipy",
        "mechkit>=0.2.6",
        # "natsort",
    ],
    # setup_requires=["pybind11>=2.3", "libcgal-dev", "libeigen3-dev"],
    # extras_require={"test": ["pytest", "natsort"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
