import setuptools

vars2find = ['__author__', '__version__', '__url__']
varstfp = {}
with open('./pept3/__init__.py') as f:
    for line in f.readlines():
        for v in vars2find:
            if line.startswith(v):
                line = line.replace(' ', '').replace("\"", '').replace("\'", '').strip()
                varstfp[v] = line.split('=')[1]

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='pept3',
    url=varstfp['__url__'],
    version=varstfp['__version__'],
    author=varstfp['__author__'],
    description='Test-time training for deep MS/MS spectrum prediction improves peptide identification',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['pept3'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=['h5py', 'numpy', 'pandas', 'torch'],
    entry_points={
        'console_scripts': [
            'pept3 = pept3:main',
        ]
    },
)
