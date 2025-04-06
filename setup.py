from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llmthinkbench",  # Package name (what people will pip install)
    version="0.1.0",  # Initial version
    author="Gaurav Srivastava",  # Replace with your name
    author_email="gauravhhh30@gmail.com",  # Replace with your email
    description="A framework for evaluating overthinking and basic reasoning capabilities of Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ctrl-gaurav/LLMThinkBench",  # Update with your repo URL
    project_urls={
        "Bug Tracker": "https://github.com/ctrl-gaurav/LLMThinkBench/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Choose appropriate license
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "vllm>=0.2.0",
        "transformers>=4.30.0",
        "numpy>=1.20.0",
        "tqdm>=4.64.0",
        "tabulate>=0.9.0",
    ],
    entry_points={
        "console_scripts": [
            "llmthinkbench=llmthinkbench.cli:main",
        ],
    },
)