from setuptools import setup, find_namespace_packages


setup(name="protosearch",
      packages=find_namespace_packages(),
      install_requires=["ase>=3.19",
                        "numpy>=1.18",
                        "CatLearn>=0.6.2",
                        "shapely"],
      package_data={
          "protosearch.oqmd": ["*.csv"],
          "protosearch.build_bulk": ["*.db", "*.dat"],
      },
      include_package_data=True,
      )
