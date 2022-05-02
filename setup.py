from setuptools import find_packages
from setuptools import setup

setup(
    name="src",
    version="0.0.1",
    maintainer="niv",
    description="ChatbotWiz",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)