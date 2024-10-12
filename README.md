# atowpy

Repository with machine learning model for Actual TakeOff Weight (ATOW) estimation. 

This repository was prepared as part of [PRC Data Challenge](https://ansperformance.eu/study/data-challenge/).
There is a solution for effectively predicting Actual TakeOff Weight variable - regression task.

## Useful literature sources

Good explanation about `taxi-out time` variable: https://www.isarsoft.com/knowledge-hub/taxi-out-time

## Solution description

This repository contains all the necessary materials, scripts, saved models and 
other artifacts needed to reproduce the solution. 

It is recommended to explore current repository with [How to work with current repository](#how-to-work-with-current-repository) section

### How to work with current repository

This repository is managed with [poetry](https://python-poetry.org/).

Project contains the following folders: 

* `atowpy` - Python module for machine learning model fitting and data processing
* `data` - folder with data 
* `examples` - scripts demonstrating how to run atowpy classes and functions
* `models` - serialized models (different versions)
* `submissions` - csv files with submissions 

In [examples folder](./examples) there are several scripts: 

1. `download_data.py` - downloading necessary data samples to local folder
2. `explore_data.py` - exploratory data analysis and auxiliary visualizations
3. `fit.py` - fitting and validating machine learning model 
4. `predict.py` - applying fitted and saved machine learning model on top of submission set

### Final solution description

TODO: add here detailed description of final most accurate solution

## Competition 

### Team information 

Team name: `team_loyal_hippo`
Team id: `d6020e5c-d553-4262-acfa-cb16ab34cc86`

### How to make a submission 

1. Check instructions about MinIO Client: https://ansperformance.eu/study/data-challenge/data.html#using-minio-client
2. Put `.exe` file in this repository in main directory
3. Configure alias `./mc.exe alias set dc24 https://s3.opensky-network.org/ ACCESS_KEY SECRET_KEY`, where `ACCESS_KEY` and `ACCESS_KEY` needed to be set
4. Use the following command (for Windows): 

```Bash
./mc.exe cp .\submissions\team_loyal_hippo_v3_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv dc24/submissions/team_loyal_hippo_v3_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv
```

Alternative command for Ubuntu:

```Bash
mc cp ./submissions/team_loyal_hippo_v3_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv dc24/submissions/team_loyal_hippo_v3_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv
```

After that according to th information from [data challenge page](https://ansperformance.eu/study/data-challenge/data.html#ranking)
wait around 30 minutes to get updated results.

> The ranking job will be automatically **run every 30 minutes** and will 
> use Root Mean Square Error (RMSE) to compare a submission for the 105959 
> flights in submission_set.csv with the (hidden) the ground truth.

Ranking page: https://ansperformance.eu/study/data-challenge/rankings.html

### Changelog

In this section, different versions of submits and models are listed, 
and models and files are paired with git commits so that each step of development
can be reproduced if necessary (see Table below).

**Note:** Before launching the model fit and predict method, do not forget to run `download_data.py` script
to download the data into data folder. And then create `models` folder if it does not exist in the repository yet.

Table 1. Model versions and corresponding submissions ([markdown_tables online generation service](https://tablesgenerator.com/markdown_tables) was used to generate table below)

| **Submission file name**                                     | **Model name** | **Commit**                                                                                   | **Description**                                                                                                                     |
|--------------------------------------------------------------|----------------|----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| team_loyal_hippo_v1_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv | model_v1.pkl   | [0f004](https://github.com/Dreamlone/atowpy/commit/0f004586ae3070c4d4df82e9820b0d9279972691) | Default sklearn random forest with numerical  features: "month", "day_of_week", "flight_duration", "taxiout_time", "flown_distance" |
| team_loyal_hippo_v2_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv | model_v2.pkl   | [4ef2b](https://github.com/Dreamlone/atowpy/commit/4ef2b071f81fa161f053e0273051e7386aa78494) | Default sklearn random forest with both numerical and basic categorical features (using one hot encoding)                           |
| team_loyal_hippo_v3_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv | model_v3.pkl   | [151a6](https://github.com/Dreamlone/atowpy/commit/151a6a0eba9f6b85bb66924ee18a1fb893423386) | Sklearn random forest with hyperperameters optimized by optuna. RMSE metric on local validation sample: **3921.87**                 |
| team_loyal_hippo_v4_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv | model_v4.pkl   | [a57e5](https://github.com/Dreamlone/atowpy/commit/a57e50ef4f4aabedbef2ebc9b10271386e7f85bf) | Sklearn random forest with extended hyperperameters optimized by optuna. Fit on the whole dataset                                   |

To get submission file with desired version, switch to commit (using for example `git reset --hard COMMIT`, where COMMIT is a commit hash), go to `examples` folder and 
launch script `predict.py` - it will generate prediction dataframe with desired file name (if the model exists).

If there is a need to fit the model first (it might happen if serialized model was too big to fit into 
git 100 MGb commit limit), launch `fit.py` file and wait until file with serialized models will appear in `models` folder. 