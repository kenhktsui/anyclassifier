from pathlib import Path
from setuptools import setup, find_packages


README_TEXT = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")


REQUIRED_PKGS = [
    "datasets>=2.20.0",
    "setfit>=1.0.3",
    "huggingface_hub>=0.23.5",
    "llama_cpp_python==0.2.84",
    "fasttext==0.9.3",
    "transformers==4.39.0"
]

setup(
    name="anyclassifier",
    version="0.1.1",
    description="One Line To Build Any Classifier Without Data",
    long_description=README_TEXT,
    long_description_content_type="text/markdown",
    maintainer="Ken Tsui",
    maintainer_email="kenhktsui@gmail.com",
    url="https://github.com/kenhktsui/anyclassifier",
    download_url="https://github.com/kenhktsui/anyclassifier/tags",
    license="Apache 2.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIRED_PKGS,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="nlp, machine learning, fewshot learning, transformers",
    zip_safe=False,  # Required for mypy to find the py.typed file
)
