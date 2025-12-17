from setuptools import setup, find_packages

setup(
    name="laser-tilt",
    version="0.1.0",
    description="Real-time section alignment + correction using ICP",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.23",
    ],
    extras_require={
        "plot": ["matplotlib>=3.7"],
    },
    entry_points={
        "console_scripts": [
            "laser-tilt-gui=laser_tilt.gui:main",
        ]
    },
)