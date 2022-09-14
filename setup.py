import setuptools


setuptools.setup(
    name="src",
    version="0.0.1",
    author="",
    author_email="",
    packages=setuptools.find_packages("src/"),
    package_dir={"src": "src"},
    description="utils et al.",
)
