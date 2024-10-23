from setuptools import find_packages, setup

setup(
    name="aroma",
    version="0.1.0",
    description="Package for Attentive ROM with Attention", 
    author="Louis Serrano",
    author_email="louis.serrano@isir.upmc.fr",
    install_requires=[
        "einops",
        "hydra-core",
        "wandb",
        "torch",
        "pandas",
        "matplotlib",
        "xarray",
        "scipy",
        "h5py",
        "timm", 
    ],
    package_dir={"aroma": "aroma"},
    packages=find_packages()
)
