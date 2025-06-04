from setuptools import setup, find_packages

setup(
    name="dlo_Ws",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description=" linear deformable object manipulation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
