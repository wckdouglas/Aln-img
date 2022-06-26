from setuptools import setup

setup(
    name="pileup_image",
    packages=[
        "pileup_image",
    ],
    install_requires=["pysam", "numpy", "matplotlib", "pydantic"],
    python_requires=">3.6.1",
)
