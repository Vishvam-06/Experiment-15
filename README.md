# Experiment-15

## Aim
To explore and apply essential data transformation techniques, specifically data normalization and categorical data encoding, using the pandas and scikit-learn libraries in Python.

## Theory
Data preprocessing is a crucial step in preparing datasets for machine learning. This experiment focuses on two primary areas: scaling numerical data and converting categorical text into a machine-readable format.

1. Data Normalization (Scaling)
Normalization adjusts the values of numerical columns in a dataset to a common scale without distorting differences in the ranges of values. This is vital for algorithms sensitive to the scale of input features (e.g., K-Nearest Neighbors, Gradient Descent based algorithms).
Min-Max Normalization: Rescales data to a fixed range, typically 0 to 1.
Formula: (X - X_min) / (X_max - X_min)
Z-score Normalization (Standardization): Transforms data to have a mean of 0 and a standard deviation of 1. It handles outliers better than Min-Max scaling.
Formula: (X - mean) / standard_deviation
Decimal Scaling: Moves the decimal point of values based on the maximum absolute value in the feature, bringing values between -1 and 1 (or 0 and 1 for positive numbers).

2. Data Encoding (Type Conversion)
Machine learning models generally require numerical input. Encoding converts categorical data (strings/text) into numerical formats.
Label Encoding: Assigns a unique integer to each category in a column. Useful for ordinal data (where order matters).
One-Hot Encoding: Creates a new binary (0 or 1) column for each category. Ideal for nominal data (where no inherent order exists) to avoid the model misinterpreting integer values as having rank.
Dummy Encoding: Similar to One-Hot Encoding but drops one of the newly created columns (using drop_first=True) to prevent the "dummy variable trap" (perfect multicollinearity), which can cause issues in regression models.

Topics Covered in the Notebook

This notebook demonstrates these concepts using both synthetic datasets created via Python dictionaries and a loaded CSV dataset (amazon_products_dataset_Expt-14.csv).
Scaling implementations:

Min-Max normalization on Price columns.

Z-score normalization on Units_Sold columns using .mean() and .std().

Simultaneous normalization of multiple columns.

Decimal scaling on Price and Reviews.

Encoding implementations:

Applying LabelEncoder to categorical features like Customer_Gender, City, and Product_Name.

Performing One-Hot Encoding using pd.get_dummies().

Implementing Dummy Encoding with drop_first=True to address multicollinearity.

## Conclusion
Through this experiment, practical techniques for standardizing numerical features and transforming categorical variables were successfully implemented. Min-Max and Z-score normalization ensured that features with varying scales contributed equally to potential model training. Furthermore, techniques like Label Encoding and One-Hot Encoding effectively converted text-based categories into numerical representations. Mastering these preprocessing steps is fundamental to building robust and accurate machine learning pipelines.
