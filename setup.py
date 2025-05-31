# setup.py
from setuptools import setup, find_packages

setup(
    name="news-inspector",
    version="0.1.0",
    # automatically find models, entities, service
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.7",
)