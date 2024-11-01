# atowpy

Team name: `team_loyal_hippo`

Team id: `d6020e5c-d553-4262-acfa-cb16ab34cc86`

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
3. `preparations_for_fitting.py` - extraction of trajectory features for flights
4. `fit.py` - fitting and validating machine learning model 
5. `predict.py` - applying fitted and saved machine learning model on top of submission set

In `atowpy/version.py` file there is an information about version of the repository. To build new model change 
`VERSION` by incrementing 1. 

### Final solution description

The best produced model according to leaderboard was version 10. 
Thus, here is described this version. To launch model prediction 
there is a need to run:

* `preparations_for_fitting.py` - script will extract features from trajectories for model
* `predict.py` - generate predictions

Model uses set of simple features (provided in "challenge_set.csv"): 

* `actual_offblock_hour` - numerical
* `arrival_hour` - numerical
* `flight_duration` - numerical
* `taxiout_time` - numerical
* `flown_distance` - numerical
* `month` - categorical
* `day_of_week` - categorical
* `aircraft_type` - categorical
* `wtc` - categorical
* `airline` - categorical

And also trajectory-based (extracted from parquet files): 

* `altitude` - numerical 
* `groundspeed` - numerical
* `u_component_of_wind` - numerical
* `v_component_of_wind` - numerical
* `latitude` - numerical
* `longitude` - numerical
* `vertical_rate` - numerical

For each flight_id, feature extraction was 
performed according to the following principle:

* Sampling - 3 seconds. So time series is consist of 3-seconds intervals 
* Time series execution started only when `altitude` difference between two neighboring time indices if bigger than 10
* Take first 30 values for each variable. Example: first 30 values of `altitude`, first 30 values of `groundspeed`

Thus, each variable extracted from parquet file for flight is represented as 30 columns (30 lags) - see picture below

![extracted_features_1.png](exploration_plots/extracted_features_1.png)

## Competition

### How to make a submission 

1. Check instructions about MinIO Client: https://ansperformance.eu/study/data-challenge/data.html#using-minio-client
2. Put `.exe` file in this repository in main directory
3. Configure alias `./mc.exe alias set dc24 https://s3.opensky-network.org/ ACCESS_KEY SECRET_KEY`, where `ACCESS_KEY` and `ACCESS_KEY` needed to be set
4. Use the following command (for Windows): 

```Bash
./mc.exe cp .\submissions\team_loyal_hippo_v12_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv dc24/submissions/team_loyal_hippo_v12_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv
```

Alternative command for Ubuntu:

```Bash
mc cp ./submissions/team_loyal_hippo_v12_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv dc24/submissions/team_loyal_hippo_v12_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv
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

| **Submission file name**                                      | **Model name** | **Commit**                                                                                   | **Description**                                                                                                                                                                                                                                                                          | **RMSE on leaderboard** |
|---------------------------------------------------------------|----------------|----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| team_loyal_hippo_v1_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv  | model_v1.pkl   | [0f004](https://github.com/Dreamlone/atowpy/commit/0f004586ae3070c4d4df82e9820b0d9279972691) | Default sklearn random forest with numerical  features: "month", "day_of_week", "flight_duration", "taxiout_time", "flown_distance"                                                                                                                                                      | -                       |
| team_loyal_hippo_v2_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv  | model_v2.pkl   | [4ef2b](https://github.com/Dreamlone/atowpy/commit/4ef2b071f81fa161f053e0273051e7386aa78494) | Default sklearn random forest with both numerical and basic categorical features (using one hot encoding)                                                                                                                                                                                | -                       |
| team_loyal_hippo_v3_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv  | model_v3.pkl   | [151a6](https://github.com/Dreamlone/atowpy/commit/151a6a0eba9f6b85bb66924ee18a1fb893423386) | Sklearn random forest with hyperperameters optimized by optuna. RMSE metric on local validation sample: **3921.87**                                                                                                                                                                      | -                       |
| team_loyal_hippo_v4_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv  | model_v4.pkl   | [a57e5](https://github.com/Dreamlone/atowpy/commit/a57e50ef4f4aabedbef2ebc9b10271386e7f85bf) | Sklearn random forest with extended hyperperameters optimized by optuna. Fit on the whole dataset                                                                                                                                                                                        | -                       |
| team_loyal_hippo_v5_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv  | model_v5.pkl   | [42508](https://github.com/Dreamlone/atowpy/commit/4250820f6b5f34316a999d9af18881c47e5da1c0) | Dask XGBoost simple model with optimization through optuna and rmse combination train 0.9 validation 0.1. RMSE metric on local validation sample: **3846.81**                                                                                                                            | -                       |
| team_loyal_hippo_v6_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv  | model_v6.pkl   | [398aa](https://github.com/Dreamlone/atowpy/commit/398aa3fb58ca3c946e00c2b3ca4dd0a96c3c104c) | Dask XGBoost simple model with optimization through optuna (with 20 iterations) and rmse combination train 0.9 validation 0.1. RMSE metric on local validation sample: **3863**                                                                                                          | 3863.07                 |
| team_loyal_hippo_v7_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv  | model_v7.pkl   | [0243f](https://github.com/Dreamlone/atowpy/commit/0243fe1149c5fdd5df5f5cc7f4e5b051c4fd2908) | Dask XGBoost trajectory models with optuna and trajectory features first 30 values with 3s sampling. Used trajectory features are altitude and groundspeed. RMSE metric on local validation sample: **3459**                                                                             | 3452.92                 |
| team_loyal_hippo_v8_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv  | model_v8.pkl   | [5d3ab](https://github.com/Dreamlone/atowpy/commit/5d3ab56b22f9ac9546d75fe71e8a9d6688c08d32) | Dask XGBoost trajectory models with optuna and trajectory features first 30 values with 3s sampling. Used trajectory features are: altitude, groundspeed, u_component_of_wind, v_component_of_wind, latitude, longitude. RMSE metric on local validation sample: **3234**                | 3215.64                 |
| team_loyal_hippo_v9_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv  | model_v9.pkl   | [f541b](https://github.com/Dreamlone/atowpy/commit/f541bccc20fa821eb1c2baa91fb600b9fd8226c8) | Dask XGBoost trajectory models with optuna and trajectory features first 30 values with 3s sampling. Used trajectory features are: altitude, groundspeed, u_component_of_wind, v_component_of_wind, latitude, longitude, vertical_rate. RMSE metric on local validation sample: **3083** | 3067.16                 |
| team_loyal_hippo_v10_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv | model_v10.pkl  | [33630](https://github.com/Dreamlone/atowpy/commit/3363048656f920af8f96572d6d4a9c6c421ee101) | Dask XGBoost trajectory model with hyperparameters and features from model v9 but fitted on a bigger dataset (95% of data)                                                                                                                                                               | 3029.57                 |
| team_loyal_hippo_v11_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv | model_v11.pkl  | [24e7c](https://github.com/Dreamlone/atowpy/commit/24e7cd7512f5681bb1c769ce894bd212128f27cd) | Dask XGBoost trajectory model with hyperparameters and features from model v10 but fitted on a bigger dataset (95% of data) and increased n estimators number to 2000                                                                                                                    | 3045.66                 |
| team_loyal_hippo_v12_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv | model_v12.pkl  | [607ab](https://github.com/Dreamlone/atowpy/commit/607ab2f149d87f150815fb7ff70549be6e0a16b6) | Mix of v10 and xgboost model with adep and ades as features and PCA for lagged timesries transformation into 4 components with application on Jan, Feb data                                                                                                                              | 3091.08                 |

To get submission file with desired version, switch to commit (using for example `git reset --hard COMMIT`, where COMMIT is a commit hash), go to `examples` folder and 
launch script `predict.py` - it will generate prediction dataframe with desired file name (if the model exists).

If there is a need to fit the model first (it might happen if serialized model was too big to fit into 
git 100 MGb commit limit), launch `fit.py` file and wait until file with serialized models will appear in `models` folder.

Note: If the model uses trajectory features, they should be extracted into csv file before model fit and before model predict - this extraction 
can be very time-consuming (several days) if the computer is not powerful enough.
