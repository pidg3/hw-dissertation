import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

# Add email into this?
# Consider licence stuff properly, need a proper separate licence file?
setup(
    name="exrt",
    version="0.1.0",
    description="Measure robustness of machine learning explanations",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/pidg3/hw-dissertation",
    author="Michael Pidgeon",
    license="GPL",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["exrt"],
    include_package_data=True,
    install_requires=["numpy"],
    entry_points={"console_scripts": ["exrt = exrt.__main__:main",]},
)
