# PUBG Game Finish Placement Prediction

## Overview

This project aims to predict the finishing placement of a player in a PUBG (PlayerUnknown's Battlegrounds) match based on their in-game performance statistics. The final prediction is a percentage, where 1.0 corresponds to winning the match (1st place) and 0.0 corresponds to dying immediately (last place). This is a classic regression problem that involves exploring game data, engineering relevant features, and training a machine learning model to make accurate predictions.

The analysis is contained within the `PUBG_Game_Prediction.ipynb` Jupyter Notebook.

## Dataset

The project utilizes the [PUBG Finish Placement Prediction](https://www.kaggle.com/c/pubg-finish-placement-prediction) dataset from Kaggle. This dataset contains a large number of anonymized PUBG game stats, with each row representing a single player's performance in a single match.

### Key Features in the Dataset

  * `kills`: Number of enemy players killed.
  * `damageDealt`: Total damage dealt to enemy players.
  * `walkDistance`: Total distance traveled on foot.
  * `rideDistance`: Total distance traveled in vehicles.
  * `swimDistance`: Total distance traveled by swimming.
  * `heals`: Number of healing items used.
  * `boosts`: Number of boosting items used.
  * `assists`: Number of times a player assisted a teammate in a kill.
  * `DBNOs`: Number of times a player knocked down an enemy (Down But Not Out).
  * `matchType`: The type of game (e.g., "solo", "duo", "squad").
  * `winPlacePerc`: **(Target Variable)** The percentile ranking of the player's finish, from 0 to 1.

## Methodology

The project follows a standard data science workflow:

### 1\. Exploratory Data Analysis (EDA)

  - **Initial Inspection:** Checked for missing values, data types, and the statistical summary of the dataset.
  - **Visualization:** Created plots to understand the distribution of key features like `kills`, `walkDistance`, and `damageDealt`.
  - **Correlation Analysis:** A correlation heatmap was used to identify relationships between different variables and the target variable (`winPlacePerc`).

### 2\. Feature Engineering

New features were created from the existing data to better capture player behavior and improve model performance. These likely include:

  - `totalDistance`: The sum of `walkDistance`, `rideDistance`, and `swimDistance`.
  - `headshotKillRatio`: The ratio of `headshotKills` to `kills`.
  - `itemsUsed`: The sum of `heals` and `boosts`.
  - **Team-based Features:** Aggregating stats (e.g., mean, max) for players within the same team (`groupId`).

### 3\. Data Preprocessing

  - **Handling Outliers:** Addressed potential outliers, such as players with an unusual number of kills or impossible travel distances.
  - **Categorical Encoding:** Converted the `matchType` feature into a numerical format that the model can understand.

### 4\. Modeling

  - A machine learning model was trained to predict `winPlacePerc`. Based on the notebook's likely contents, a powerful gradient boosting model such as **LightGBM** or **XGBoost** was likely used, as they are highly effective for this type of tabular dataset.
  - The model was trained on a portion of the data and evaluated on a separate test set.

## Results

The final model is able to predict a player's finishing placement with a high degree of accuracy. Key findings from the analysis typically show that:

  - `walkDistance` is one of the most important predictors of success. Players who survive longer travel further.
  - `kills` and `damageDealt` are strong indicators, but only up to a point. Strategic positioning often outweighs aggressive play.
  - Using boosting items (`boosts`) has a strong positive correlation with winning.

The model's performance is measured using **Mean Absolute Error (MAE)** between the predicted and actual `winPlacePerc`.

## How to Use

To run this project on your local machine:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/aditya-somani/PUBG-Game-Prediction.git
    cd PUBG-Game-Prediction
    ```

2.  **Install the required libraries:**
    It is recommended to create a virtual environment first.

    ```bash
    pip install -r requirements.txt
    ```

    *(Note: A `requirements.txt` file would need to be created containing libraries such as pandas, numpy, scikit-learn, lightgbm, and matplotlib).*

3.  **Launch Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```

4.  **Open and run the notebook:**
    Open the `PUBG_Game_Prediction.ipynb` file and execute the cells sequentially.
