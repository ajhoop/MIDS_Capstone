# mPowerSmallBiz

This repo contains the working files for the UC Berkeley Master of Information and Data Science (MIDS) Capstone (W210) project. The team developing this project is Ammara Essa, Andy Hoopengardner, Shaji Kunjumohamed, and Padma Sridhar. The final presentation for this course is included in this repo.

## Overview

mPowerSmallBiz is a marketplace that links together US small businesses with potential buyers and buyers with new sources of supply. Establishing these new connections is especially important in 2021 for several reasons.

* COVID-19 has caused severe supply chain disruptions that have rippled across sectors and negatively impacted US businesses. Total seasonally-adjusted US imports of goods declined from $2.5T to $2.1T in 2020. Many companies are devoting serious time, energy, and resources into reenforcing their supply chains.
* COVID-19 negatively impacted US small businesses, with revenue down an average of 20% during the first eight months of 2020. The $940B in loans available through the Paycheck Protection Program (PPP) have helped many businesses survive, but are not likely to be a long-term solution.

The mPowerSmallBiz marketplace is powered by a specialized search engine trained using natural language processing (NLP). The search engine takes in free text product descriptions and predicts the appropriate Harmonized System, or HS Code. HS Codes are an international standard for product classification that is administered by the World Customs Organization. The codes are hierarchical, with the first two digits designating the broad category of item and contain a minimum of six digits allowing for increased specificity. In 2019, there were over 4,200 different HS codes imported into the United States.

Once a product description has been matched to an HS Code, the mPowerSmallBiz marketplace can identify US businesses that have imported similar items based on our dataset of over 12M US Customs import records from 2019. The marketplace can also identify US small businesses capable of producing similar items by matching the HS Code to a North American Industry Classification System (NAICS) Code and then to US businesses that received PPP loans. These results are returned via an interactive interface that enables the user to explore and identify new leads, whether they are buying or selling.

The MVP website is available here: http://mpowersmallbiz.com

## Repo Organization

This github repo has three main areas:

* Data
* Notebooks
* SRC

### Data

The data portion of the repo describes each of the datasets used, which include a combination of publicly available, government, and commercially procured datasets. The datasets were stored in an Amazon S3 bucket for the duration of this project and are not re-produced in this repo. Where possible, we provide links to the sites from which the datasets were downloaded.

### Notebooks

This section of the repo contains the various notebooks the team used to perform EDA and model development. During the project, each team member worked independently and shared code as needed. The "Dataprep" directory contains the code used to create the various samples used to test different models, as well as to perform some basic data preparation and de-duplication.

### SRC

This section of the repo contains our final model.

## Tools
<p align="center">
  <img width="460" height="300" src="/images/tools.PNG">
</p>


