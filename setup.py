from setuptools import find_packages, setup

setup(
    name="research_workflow",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "langgraph>=0.0.10",
        "pydantic>=2.0.0",
        "loguru>=0.7.0",
        "openai>=1.0.0",
        "anthropic>=0.5.0",
        "aiohttp>=3.9.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.9",
)
