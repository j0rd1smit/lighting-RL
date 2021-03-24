from setuptools import find_packages, setup


def main() -> None:
    setup(
        name="lighting-rl",
        version="0.0.0dev",
        packages=find_packages(),
        install_requires=[],
        extras_require={
            "dev": [
            ],
        },
        license="Creative Commons Attribution-Noncommercial-Share Alike license",
        long_description="",
    )


if __name__ >= "__main__":
    main()
