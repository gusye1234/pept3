from importlib_metadata import entry_points
import setuptools
import stums

with open("README.md", "r") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as fh:
    deps = [l.strip() for l in fh]

setuptools.setup(
    name="stums",
    url=stums.__url__,
    version=stums.__version__,
    author=stums.__author__,
    description="Semi-supervised Fine-tuning for Mass Spectrum Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['stums'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=deps,
    entry_points={
        "console_scripts": [
            "stums = stums:main",
        ]
    }
)
