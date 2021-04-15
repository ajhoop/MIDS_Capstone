# mPowerSmallBiz

This repo contains the working files for the UC Berkeley Master of Information and Data Science (MIDS) Capstone (W210) project. The team developing this project is Ammara Essa, Andy Hoopengardner, Shaji Kunjumohamed, and Padma Sridhar.

## Overview

mPowerSmallBiz is a marketplace that links together US small businesses with potential buyers and buyers with new sources of supply. Establishing these new connections is especially important in 2021 for several reasons.

* COVID-19 has caused severe supply chain disruptions that have rippled across sectors and negatively impacted US businesses. Total seasonally-adjusted US imports of goods declined from $2.5T to $2.1T in 2020. Many companies are devoting serious time, energy, and resources into reenforcing their supply chains.
* COVID-19 negatively impacted US small businesses, with revenue down an average of 20% during the first eight months of 2020. The $940B in loans available through the Paycheck Protection Program (PPP) have helped many businesses survive, but are not likely to be a long-term solution.

The mPowerSmallBiz marketplace is powered by a specialized search engine trained using natural language processing (NLP). The search engine takes in free text product descriptions and predicts the appropriate Harmonized System, or HS Code. HS Codes are an international standard for product classification that is administered by the World Customs Organization. The codes are hierarchical, with the first two digits designating the broad category of item and contain a minimum of six digits allowing for increased specificity. In 2019, there were over 4,200 different HS codes imported into the United States.

Once a product description has been matched to an HS Code, the mPowerSmallBiz marketplace can identify US businesses that have imported similar items based on our dataset of over 12M US Customs import records from 2019. The marketplace can also identify US small businesses capable of producing similar items by matching the HS Code to a North American Industry Classification System (NAICS) Code and then to US businesses that received PPP loans. These results are returned via an interactive interface that enables the user to explore and identify new leads, whether they are buying or selling.

## Setting up the src code and customizing the conda environment

The conda environment in `environment.yml` is a starter environment, only for setting up the repo. I include instructions for renaming and customizing the conda environment after the initial installation is complete. Use the following steps to complete the installation of the environment and make code stored in src available as a package.

After cloning the repo, navigate into the repo and run:

```
# create the conda environment
conda env create -f environment.yml

# activate the conda environment
conda activate initial-env
```

Then, to make a copy of the initial-env conda environment and rename it to whatever you want to use as your custom environment name, use the following, replacing `custom-env` with your preferred name.

```
# first make a copy of initial-env 
# (can use the flag --offline if you don't want to redownload packages)
conda create --name custom-env --clone initial-env

# second, delete initial-env
conda remove --name initial-env --all
```

Now you have your own custom conda environment. I suggest running the following lines to make your environment available to Jupyter as a kernel and export your new environment to the `environment.yml` file (remember to rerun the latter as you add more libraries to your environment).

```
# make this conda environment available as a kernel in jupyter
python -m ipykernel install --user --name custom-env --display-name "custom-env"

# export environment.yml file
conda env export > environment.yml
```

If you for some reason you already have a conda environment you want to use with this repo, all you'll need to do is navigate into the repo, activate your environment, and then run `pip install -e .` to set up the src folder.
