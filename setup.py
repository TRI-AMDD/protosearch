from setuptools import setup, find_namespace_packages

DESCRIPTION = "Software for enumerating crystal structure prototypes to by used for active " \
              "learning exploration with DFT and machine learning."

LONG_DESCRIPTION = """
Protosearch is software that supports the creation of libraries of
crystal structures according to structure prototypes derived from
the ICSD and OQMD
"""

setup(
    name='protosearch',
    url="https://github.com/ToyotaReseachInstitute/protosearch",
    version="2020.5.10",
    packages=find_namespace_packages(),
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    install_requires=["ase>=3.19",
                      "numpy>=1.18",
                      "CatLearn>=0.6.2",
                      "shapely"],
    package_data={
        "protosearch.oqmd": ["*.csv"],
        "protosearch.build_bulk": ["*.db", "*.dat"],
    },
    include_package_data=True,
    extras_require={
        "enumerator": ["BulkEnumerator>=0.0.2"],
    },
    classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
    ],
    # TODO: Edit as needed
    author="Kirsten Winther, SUNCAT-Center, TRI-AMDD",
    author_email="",
    maintainer="Kirsten Winther",
    maintainer_email="",
    license="Apache",
    keywords=[
        "materials", "chemistry", "science", "crystal",
        "density functional theory", "energy", "AI", "artificial intelligence",
        "sequential learning", "active learning"
    ],
    )