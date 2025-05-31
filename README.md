# Sentiment_analysis_using_yelp_dataset
=======
# Sentiment Analysis Project

This repository contains a sentiment analysis project implemented using PySpark. The project analyzes user reviews from the Yelp dataset to determine their sentiment (positive, negative, or neutral).

## Project Structure

- `Sentiment Analysis.py`: Main script for performing sentiment analysis.
- `results_dashboard.py`: Script for visualizing the results of the sentiment analysis.
- `metrics/`: Contains evaluation metrics and visualizations.
  - `accuracy.txt`: Accuracy of the models.
  - `model_comparison.png`: Comparison of different models.
  - `sentiment_distribution.png`: Distribution of sentiments in the dataset.
- `models/`: Contains trained models for sentiment analysis.
  - `logistic_regression_model/`: Logistic Regression model.
  - `naive_bayes_model/`: Naive Bayes model.
  - `random_forest_model/`: Random Forest model.
- `yelp_academic_dataset_review.json`: The Yelp dataset used for training and testing (not included in this repository).

## Requirements

To run this project, install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. Download the Yelp dataset (`yelp_academic_dataset_review.json`) from the [Yelp Dataset Challenge](https://www.yelp.com/dataset).
2. Place the dataset in the root directory of this project.
3. Run the `Sentiment Analysis.py` script to perform sentiment analysis.
4. Use `results_dashboard.py` to visualize the results.

## Results

The results of the sentiment analysis are stored in the `metrics/` directory. This includes accuracy metrics, model comparisons, and sentiment distribution visualizations.

## Models

The trained models are stored in the `models/` directory. You can use these models to predict sentiments on new data.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
