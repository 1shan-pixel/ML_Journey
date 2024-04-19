# for missing values 
Missing completely at random (MCAR): The missing data entries are random and don't correlate with any other data.
Missing at random (MAR): The missing values depend on the values of other variables.
Missing not at random (MNAR): The missing values have a particular pattern or logic.

# Strategies to Handle Missing Data
Armed with the knowledge of missing data and its types, it's time to decide how to handle them. Broadly, you can consider three main strategies:

Deletion: This involves removing the rows and columns containing missing data. However, this might lead to the loss of valuable information.
Imputation: This includes filling missing values with substituted ones, like the mean, median, or mode (the most common value in the data frame).
Prediction: This involves using a predictive model to estimate the missing values.

# Outlier Detection 

The three common methods for outlier detection are Z-score (identifying data points with a Z-score greater than 3 as outliers), IQR (defining outliers as observations outside the range of (q1−1.5⋅iqr) and (q3 + 1.5. iqr) , and Standard Deviation (categorizing data points more than three standard deviations from the mean as outliers).

# Handling Outliers
After identifying outliers, we'll need to decide on the best strategy for handling them, such as:

Dropping: If the outlier does not add valuable information or is significantly skewing our data, one option to consider is dropping the outlier.
Capping: We could also consider replacing the outlier value with a certain maximum and/or minimum value.
Transforming: Techniques such as log transformations are especially effective when dealing with skewed data. This type of transformation can reduce the impact of the outliers.


# Normalization and Standardization 

Normalization keeps the data range within a certain range generally between 0 and 1, standardization ensures the mean to be zero and the standard deviation to be 1. 


Choose normalization when your data needs to be bounded within a specific range (0 to 1, for example) and is not heavily influenced by outliers. This is particularly useful for algorithms that are sensitive to the scale of the data, such as neural networks and k-nearest neighbors. On the other hand, standardization is more effective when your data has a Gaussian distribution, and you are dealing with algorithms that assume this, such as linear regression, logistic regression, and linear discriminant analysis.


# Introduction to Feature Engineering 

Feature engineering is the process of creating optimized features that improve the effectiveness of machine learning algorithms. This process utilizes the data to create new features and modify existing ones. This might involve creating new features, transforming existing features, or identifying and removing irrelevant ones. For instance, in our Titanic dataset, we have properties or indicators like age, sex, pclass, etc., which might need some optimizing.

# Introduction to Linear Regression

Linear Regression is a prime tool in a statistician's toolbox, intended to decipher the relationship between two or more variables. To break it down into simpler terms, imagine you are an explorer observing a sunrise. You realize that, the higher the sun rises in the sky, the brighter it gets. This scenario is a simple example of linear regression, where the height of the sun (an independent variable) and the brightness (a dependent variable) share a linear relationship.

# Understanding Logistic Regression
Logistic Regression is your go-to statistical model for binary classification tasks. For example, if you need to classify objects, such as distinguishing an apple from an orange based on features like color or size, logistic regression can do the job. Don't let the term 'regression' in its name mislead you, though. Unlike linear regression that predicts a continuous outcome, logistic regression works in the realm of probabilities, making it superb for dichotomous outcomes, like yes/no or true/false decisions.


# Decision Trees 


# Cross Validation 

In Cross-Validation, we divide our dataset into 'K' parts, or folds. We then train our model 'K' times, each time using a different fold as our Test Set. This yields 'K' performance scores, which we average to get a final score

# Performance Metrics 

For continous value predictions we use metrics like mean absolute error mean squared error and even root mean squared error. In the case of categorical data , we use thigns like precision accuracy , recall and F1 score for the machine learning model. 

Here , harmonic mean of accuracy and recall is the F1 score. 


# Strengths and Limitations of Linear Regression

Being aware of a model's strengths and weaknesses allows us to be mindful of its suitability for addressing specific problems. In the case of Linear Regression:

Strengths of Linear Regression:

Simplicity: It is easy to comprehend and implement.
Speed: It has quicker computation than some other models.
Handles continuous data well: It can model the relationship between continuous features and outputs.

Limitations of Linear Regression:

Sensitivity to extreme values: A single outlier can significantly alter the model.
Infers linear relationships: It assumes a simple linear correlation, which might not always hold true in real-world data.
Cannot model complex patterns: Models that can capture complex data relationships perform better.

Strengths and Limitations of Logistic Regression
Like Linear Regression, Logistic Regression also has its own set of strengths and constraints.

# Strengths of Logistic Regression:

Handles categorical data: It's adept at modeling problems with a categorical target variable.
Provides probabilities: It helps in understanding the level of certainty of the predictions.
Offers solid inference: Insights into how each feature affects the target variable can be feasibly deduced.

Limitations of Logistic Regression:

Inefficient with complex relationships: It has trouble capturing complex patterns.
Assumes linear decision boundaries: It might not always align with complex dataset structures.
Unsuitable for continuous outcomes: Due to its probabilistic approach, it does not provide continuous outputs.


# Strengths and Limitations of Decision Trees

Decision Tree models also have unique abilities and setbacks.

Strengths of Decision Trees:

Transparent: They are easy to understand and interpret.
Handles categorical and numerical data: They can conveniently work with a mix of categorical and numerical features.
Can capture complex patterns: They are capable of fitting highly complex datasets.
Limitations of Decision Trees:

Prone to overfitting: They might create an overly complex model that does not generalize well.
Sensitivity to data tweaks: Small changes in data could lead to different trees.
Biased for the dominating class: If one class outnumbers other classes, the decision tree might create a biased tree.

# Optimizing Models 

GridSearchCV and RandomizedSearchCV are techniques used by sklearn for hyperparameter tuning, which is the process of finding the ideal parameters (those knobs and switches we talked about) that give our model the best performance. As its name suggests, GridSearchCV exhaustively tries out different combinations of parameter values (provided by you) and delivers the best combination. RandomizedSearchCV, on the other hand, selects random parameter combinations from a distribution of values provided. Depending on the precision required and time at hand, we choose one over the other.


# Feature Engineering

Feature Engineering, in its simplest terms, encompasses the techniques used to create new, transformative algorithms that extract more meaningful information from raw data, thereby increasing the predictive power of machine learning or statistical modeling. Not only does it significantly enhance a model's predictive capability, understanding and implementing feature engineering can also lead to decidedly more efficient models.


# PCA 

There are various approaches to reduce dimensionality, including feature selection strategies, where you selectively use a subset of features, and feature extraction or transformation methods, that create a more manageable set of new 'composite' features. One popular feature extraction method is Principal Component Analysis (PCA).



# Different Methods of Feature Selection 

Now let's deep-dive into the primary categories of feature selection algorithms: Filter Methods, Wrapper Methods, and Embedded Methods. In each case, we will supplement our analysis with illustrative Python code, using the UCI's Abalone Dataset.


# Understanding Feature Interaction
In a machine-learning context, feature interaction can be divided into additive and multiplicative interactions. An additive interaction means that the effects of two or more individual features combine, contributing to the target variable. Conversely, multiplicative interaction implies that features enhance or dampen each other's impact.























# Things to note 

Use biased standard deviation , ssof = 1 when sample sizes are very small . 

Remember MinMaxScaler and others expect the data to be in a 2d array always make sure to do [[]]

use "" if something else contains ''
overfitting is where a model performs great on the data it is trained on and poorly on unseen data

If your training accuracy is much higher than your test accuracy, it might be a sign that the model is overfitting to the training data. Conversely, if both accuracies are low, the model might be under-fitting.

Detailed features can bring out underlying patterns that can further enlighten the model. 