import setuptools

vars2find = ['__author__', '__version__', '__url__']
varstfp = {}
with open("./tfpms/__init__.py") as f:
    for line in f.readlines():
        for v in vars2find:
            if line.startswith(v):
                line = line.replace(" ", '').replace(
                    "\"", '').replace("\'", '').strip()
                varstfp[v] = line.split('=')[1]

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='tfpms',
    url=varstfp['__url__'],
    version=varstfp['__version__'],
    author=varstfp['__author__'],
    description='Data-specific model fine-tuning for MS/MS spectrum prediction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['tfpms'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=["h5py",
                      "numpy",
                      "pandas",
                      "torch"
                      ],
    entry_points={
        'console_scripts': [
            'tfpms = tfpms:main',
        ]
    },
)
