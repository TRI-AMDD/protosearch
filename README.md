# protosearch
Software for active learning of new materials from DFT and crystal structure prototypes

## Installation
Note that the required BulkEnumeration is currently available by request only.


## TODO:
* Revisit fingerprinting scheme
  - The indices between the `prototypes` db table and the `systems` don't agree with one another
    - The `systems` table excludes entries from the `prototypes` table (max_atoms)
    - Consider creating a standardized id system from the outset that is shared among all tables

  - Fingerprints must be standardized
  - Fingerprints must be updated every time a new entry is added since standardization depends on the "mean" and/or standard deviation of the whole data set

## Required Data Files
Large data files which are necessary to run certain modules (OQMD db interface) are hosted on the TRI s3 bucket in the following location:

s3://matr.io/camd/shared-data/protosearch-data

The OQMD data file is in the following location:

s3://matr.io/camd/shared-data/protosearch-data/materials-db/oqmd/oqmd_ver3.db

To download run the following command
`aws s3 cp s3://matr.io/camd/shared-data/protosearch-data/materials-db/oqmd/oqmd_ver3.db .`
