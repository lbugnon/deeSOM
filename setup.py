import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deeSOM",
    version="0.1",
    author="L.A. Bugnon, C. Yones, D. Milone, G. Stegmayer",
    author_email="lbugnon@sinc.unl.edu.ar",
    description="Deep ensemble-elastic self-organized map (deesom): a SOM based classifier to deal with large and highly imbalanced data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lbugnon/deesom",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
) 
