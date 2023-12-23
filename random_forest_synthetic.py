import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('/Users/meeps360/cs158-final-project/week_approach_maskedID_timeseries.csv')

# Separate the dataset into injured and uninjured groups
injured_samples = df[df['injury'] == 1]
uninjured_samples = df[df['injury'] == 0]

# Use SMOTE to generate synthetic samples for the minority class
x = df.drop(columns=['injury', 'Athlete ID', 'Date'])
y = df['injury']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(x, y)

# Split the resampled data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train a model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

#testing original dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

#create a table showing trends in what makes a player more likely to be injured
#showing the top 10 features with the highest feature importance and label columns
feature_imp = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

#label the columns
feature_imp.columns = ['Feature', 'Importance']
print(feature_imp.head(10))
#what was the least important feature?
print(feature_imp.tail(1))

#export the table to a csv
feature_imp.to_csv('feature_imp.csv')

# #create a graph showing the top 10 features with the highest feature importance
# # Creating a bar plot
# sns.barplot(x=feature_imp[:10], y=feature_imp[:10].index)
# # Add labels to your graph
# plt.xlabel('Feature Importance Score')
# plt.ylabel('Features')
# plt.title("Visualizing Important Features")
# plt.legend()
# plt.show()
# plt.savefig('feature_imp.png')

#what are all the features?
print(feature_imp.index)