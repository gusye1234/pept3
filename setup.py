import setuptools
from importlib_metadata import entry_points

import tfpms

with open('README.md', 'r') as fh:
    long_description = fh.read()
with open('requirements.txt', 'r') as fh:
    deps = [l.strip() for l in fh]

setuptools.setup(
    name='tfpms',
    url=tfpms.__url__,
    version=tfpms.__version__,
    author=tfpms.__author__,
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
    install_requires=deps,
    entry_points={
        'console_scripts': [
            'tfpms = tfpms:main',
        ]
    },
)
