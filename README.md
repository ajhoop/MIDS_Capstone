# mPowerSmallBiz

This repo contains the working files for the UC Berkeley Master of Information and Data Science (MIDS) Capstone (W210) project. The team developing this project is Ammara Essa, Andy Hoopengardner, Shaji Kunjumohamed, and Padma Sridhar.

## Overview

mPowerSmallBiz is a marketplace that links together US small businesses with potential buyers and buyers with new sources of supply. Establishing these new connections is especially important in 2021 for several reasons.

* COVID-19 has caused severe supply chain disruptions that have rippled across sectors and negatively impacted US businesses. Total seasonally-adjusted US imports of goods declined from $2.5T to $2.1T in 2020. Many companies are devoting serious time, energy, and resources into reenforcing their supply chains.
* COVID-19 negatively impacted US small businesses, with revenue down an average of 20% during the first eight months of 2020. The $940B in loans available through the Paycheck Protection Program (PPP) have helped many businesses survive, but are not likely to be a long-term solution.

The mPowerSmallBiz marketplace is powered by a specialized search engine trained using natural language processing (NLP). The search engine takes in free text product descriptions and predicts the appropriate Harmonized System, or HS Code. HS Codes are an international standard for product classification that is administered by the World Customs Organization. The codes are hierarchical, with the first two digits designating the broad category of item and contain a minimum of six digits allowing for increased specificity. In 2019, there were over 4,200 different HS codes imported into the United States.

Once a product description has been matched to an HS Code, the mPowerSmallBiz marketplace can identify US businesses that have imported similar items based on our dataset of over 12M US Customs import records from 2019. The marketplace can also identify US small businesses capable of producing similar items by matching the HS Code to a North American Industry Classification System (NAICS) Code and then to US businesses that received PPP loans. These results are returned via an interactive interface that enables the user to explore and identify new leads, whether they are buying or selling.

## Repo Organization

This github repo has three main areas:

* Data
* Notebooks
* SRC

### Data

This project uses


