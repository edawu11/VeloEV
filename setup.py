from setuptools import setup, find_packages

setup(
    name="veloev",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        'matplotlib>=3.10.0',
        'numpy>=2.0.0',
        'pandas>=2.3.0',
        'scanpy>=1.11.0',
        'scikit-learn>=1.5.0',
        'scipy>=1.10.0',
        'scvelo>=0.3.0',
        'seaborn>=0.13.0',
        'tqdm>=4.60.0'
    ],
    author="Yida Wu",
    author_email="yidawu@cuhk.link.edu.cn",
    description="Evaluation and visualization for RNA velocity methods",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/edawu11/veloev",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)