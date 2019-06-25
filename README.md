# TEMP | RF | 190606
# protosearch
Software for active learning of new materials from DFT and crystal structure prototypes

## Installation
Note that the required BulkEnumeration is currently available by request only.


## Required Data Files
Large data files which are necessary to run certain modules (OQMD db interface) are hosted on the TRI s3 bucket in the following location:

s3://matr.io/camd/shared-data/protosearch-data

The OQMD data file is in the following location:

s3://matr.io/camd/shared-data/protosearch-data/materials-db/oqmd/oqmd_ver3.db

To download run the following command
`aws s3 cp s3://matr.io/camd/shared-data/protosearch-data/materials-db/oqmd/oqmd_ver3.db .`
