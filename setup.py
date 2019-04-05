from setuptools import setup, find_packages
import warnings


# TODO: resolve dependency on BulkEnumeration
setup(name="protosearch",
      packages=find_packages(),
      install_requires=["ase>=3.17",
                        "numpy>=1.14",
                        # "BulkEnumeration>=0.2"
                        ],
      # dependency_links=["https://gitlab.com/ankitjainmeiitk/Enumerator.git"]
      )

if __name__ == '__main__':
    try:
        import bulk_enumerator as be
    except ImportError:
        warnings.warn("BulkEnumeration is required for proper functioning"
                      "of protosearch, please request the code.")
