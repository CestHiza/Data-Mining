# %% [markdown]
# # Project 2
# ## Introduce the problem
# ### What is the problem you are trying to solve?
# ##### Predicting `Titanic` survival
# ### What questions are you trying to find answers to?
# ##### What factors have contributed to `Titanic` survival?

# %%
# dependencies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# %%
titanic_data = pd.read_csv(r"C:\Users\cesth\Downloads\titanic.csv")

# Print the dataset
print(titanic_data.head())

# %%
# What data types are in the dataset?
print(titanic_data.dtypes)

# %%
# What are the features, columns and data in the dataset?
print(titanic_data.info())

# %%
# Check the missing values
print(titanic_data.isnull().sum())

# %% [markdown]
# ## Visualizations

# %%
# Vizualize the age distribution
plt.hist(titanic_data['Age'], bins=20)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution')
plt.show()

# %%
# Plot the gender distribution
plt.hist(titanic_data['Sex'], bins=2)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Distribution')
plt.show()

# %%
# Plot the class distribution
plt.hist(titanic_data['Pclass'], bins=3)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()

# %% [markdown]
# ## Preprocessing

# %%
# Fill in the missing values in the Age column
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)

# Fill in the missing values in the Embarked column
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# %%
# Again, plot age distribution
plt.hist(titanic_data['Age'], bins=20)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution')
plt.show()

# %% [markdown]
# ##### Notice how the age distribution changed

# %% [markdown]
# ## Train the Model

# %%
# Convert integer values in 'Survived' column to boolean values
titanic_data['Survived'] = titanic_data['Survived'].astype(bool)

# Lets double check the data type
print(titanic_data['Survived'].dtype)


# %%
# Feature selection
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X = titanic_data[features]
y = titanic_data['Survived']

# Convert categorical variables to indicator variables
X = pd.get_dummies(X, drop_first = True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# %%
# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on test set
y_pred = rf_classifier.predict(X_test)


# %%
# Evaluate the Model
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# %% [markdown]
# ## `Vizualization+`

# %%
import seaborn as sns

# Histogram of Age
plt.figure(figsize = (8, 6))
sns.histplot(titanic_data['Age'], bins = 20, kde = True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# %%
# Bar plot of Sex
plt.figure(figsize = (8, 6))
sns.countplot(x = 'Sex', data = titanic_data)
plt.title('Passenger Count by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# %%
# Bar plot of survival status
plt.figure(figsize = (8, 6))
sns.countplot(x = 'Survived', data = titanic_data)
plt.title('Survival Count')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# %%
# Bar plot of survival by Sex
plt.figure(figsize = (8, 6))
sns.countplot(x = 'Sex', hue = 'Survived', data = titanic_data)
plt.title('Survival by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title = 'Survived', loc = 'upper right')
plt.show()

# %%
# Scatter plot of Age v. Fare, colored by Survived
plt.figure(figsize = (8, 6))
sns.scatterplot(x = 'Age', y = 'Fare', hue = 'Survived', data = titanic_data)
plt.title('Age vs Fare by Survival')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()


# %%
# Factor plot of survival by Pclass
plt.figure(figsize = (8, 6))
sns.catplot(x = 'Pclass', hue = 'Survived', kind = 'count', data = titanic_data)
plt.title('Survival by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()

# %%
# Pair plot of numerical features
sns.pairplot(titanic_data[['Age', 'Fare', 'SibSp', 'Parch', 'Survived']], hue = 'Survived', diag_kind = 'kde')
plt.show()


