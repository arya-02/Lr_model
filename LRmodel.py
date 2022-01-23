
#importing the libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#reading the data
data = pd.read_csv("50_Startups.csv")

#selecting all rows and all cols except last one for input variables
x = data.iloc[:,:-1].values
#last col for output variable
y = data.iloc[:,4].values

#transforming the categorical data in col 3 
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
x[:,3] = label.fit_transform(x[:,3])

#splitting,training and predicting
regr = LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
regr.fit(x_train,y_train)
pred=regr.predict(x_test)

#calculating accuracy
score=r2_score(y_test,pred)
print(score*100)

