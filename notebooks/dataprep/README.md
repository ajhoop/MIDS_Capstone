# Notebooks - Data Preparation

This directory contains notebooks for data pre-processing.

## Notebook Descriptions

* __part1_us_imports_bol_eda__: processes sample 2015-2020 us import data from shipping carriers. Data not included.
* __part2_us_customs_2019_data_prep_sampling_FULLDESC__: processes 2019 US customs imports data through data cleanup and enhancements. Shipping records with multiple HS codes per container are exploded to unique rows. Sample sets are created for bike parts and HS chapters 39 and 40. Data not included.
* __part3_us_customs_2019_data_prep_sampling_remove_multiple_hscode_FULLDESC__: processes 2019 US customs imports data through data cleanup and enhancements. Shipping records with multiple HS codes per container are dropped in this version. Data saved into parquet files by month. Generic code written to create data samples for any HS code or HS code chapter. Data not included.
* __part4_sba_ppp_data_processing__: process the PPP dataset downloaded from SBA website (2020-Feb2021). Data not included.

## Data Sources

This project used data from a variety of public, government, and commercial sources. We are not storing datafiles in our repo, but provide links below for datasets that are publicly available.

* 2019 US Customs Import Data. This dataset was our primary training dataset for the NLP models. We procured this dataset from ManifestDB (https://www.manifestdb.com) and are not able to share a copy under the terms of sale. We also used this dataset to build our database of US buyers.
* 2015 - 2020 US Bill of Lading Data. This was a longitudinal sample of Bills of Lading for imports and exports for the US. We procured this dataset from ManifestDB, as well, under similar terms. As we dug into this dataset, we learned that it was a small subset of total imports and we were unable to determine how the samples were drawn (and thus, how the dataset was biased) so we did not ultimately use this dataset for our final project.
* HS Code Definitions. The definitions for the Harmonized System (HS) Codes were downloaded from the US International Trade Commission and are available at https://www.usitc.gov/tata/hts/archive/index.htm.
* Paycheck Protectin Program Data. This dataset is provided by the US Government's Small Business Administration (SBA) and is available for download at https://www.sba.gov/funding-programs/loans/covid-19-relief-options/paycheck-protection-program/ppp-data#section-header-2. We used this dataset to build our database of US small businesses.
* Data from the US Census Bureau includes a flag for "rural" or "urban". This dataset is available here: https://github.com/Ro-Data/Ro-Census-Summaries-By-Zipcode . We used this to create the flag for urban or rural for each entry in our supplier and buyer database.
