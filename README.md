# Experiment-15

## Aim
To perform advanced data preprocessing techniques, specifically focusing on Data Normalization (feature scaling) and Data Encoding (converting categorical data to numerical data) using the Python pandas and scikit-learn libraries.

## Theory
Real-world datasets often contain features with highly varying magnitudes, ranges, and data types (both numerical and text). To feed this data into machine learning algorithms effectively, it must be scaled and converted into a standardized numerical format. This experiment covers two primary preprocessing phases:

1. Data Normalization (Feature Scaling)
Normalization adjusts the numerical values of different columns to a common scale without distorting differences in the ranges of values.

Min-Max Normalization: Rescales the data so that all values fall within a specific range, typically [0, 1]. It is calculated as (x - min) / (max - min).

Z-Score Normalization (Standardization): Centers the data around a mean of 0 with a standard deviation of 1. It handles outliers better than Min-Max and is calculated as (x - mean) / standard_deviation.

Decimal Scaling: Moves the decimal point of values of feature x. The number of decimal points moved depends on the maximum absolute value of x.

2. Data Encoding
Machine learning models require input and output variables to be numeric. Encoding is the process of converting categorical (text) data into numerical formats.

Label Encoding (LabelEncoder): Converts each category value into a unique integer (e.g., City names like 'Pune' and 'Mumbai' become 4 and 3). Useful for ordinal data but can imply false relationships in nominal data.

One-Hot Encoding (pd.get_dummies): Creates new binary (True/False or 1/0) columns for each unique category in the original column. This prevents the model from assuming a natural ordering between categories.

Dummy Encoding (drop_first=True): Similar to One-Hot Encoding but drops the first resulting column to avoid multicollinearity (the "dummy variable trap"), resulting in N-1 binary columns for N categories.

## Conclusion
In this experiment, data normalization and categorical data encoding were successfully implemented using pandas and sklearn.preprocessing. Numerical features such as Price, Units Sold, and Discount were effectively scaled using Min-Max, Z-score, and Decimal scaling methodologies to ensure uniform feature magnitude.
