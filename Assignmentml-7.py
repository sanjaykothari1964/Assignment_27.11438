

#Importing all the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import dataset
nba = pd.read_csv("nba_2013.csv")

#Dropiing the last two unnecessary columns
nba = nba.iloc[:,1:29]

#Drop brief_team_Id column too
nba = nba.drop("bref_team_id", axis =1)



#Assigning the ddependant and independant variables
X = nba.iloc[:,:-1].values
y = nba.iloc[:,-1].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,[7,10,13,14,17]])
X[:,[7,10,13,14,17]] = imputer.transform(X[:,[7,10,13,14,17]])


#Label encoding categorical value
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

#Implementing dummy trap
X= X[:,1:]


#Splitting the training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Applying Algorithm KNN to the training dataset
from sklearn.neighbors import KNeighborsRegressor
classifier = KNeighborsRegressor(n_neighbors=5, metric= "minkowski", p =2)
classifier.fit(X_train, y_train)

#Predicting the test data
y_pred = classifier.predict(X_test)

#Plotting the scatter plots
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Pts")
plt.ylabel("Predicted pts")
plt.title("Actual Pts vs Predicted Pts")


