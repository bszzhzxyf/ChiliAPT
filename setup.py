from setuptools import setup, find_packages

setup(
    name="chiliapt",
    version="0.1.0",
    author="Yifei Xiong",
    author_email="xiongyf@shao.ac.cn",
    description="Simple proposal tool for Chili IFU",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/chiliapt",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "matplotlib",
        "shapely",
        "astroquery",
        "astropy",
        "pandas",
    ],
)