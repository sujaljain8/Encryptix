
import numpy as np  #linear algebra
import pandas as pd   #data processing

df =pd.read_csv("Iris.csv")

df

df.shape

"""## Dropping the Column"""

df=df.drop(columns=["Id"])

df

df.head() #Return first 5 entries

"""## Transforming Categorical Data into Numeric Data"""

df["Species"].replace({"Iris-setosa":1,"Iris-versicolor":2,"Iris-virginica":3},inplace=True)

df

"""## Creating Arrays"""

x=pd.DataFrame(df,columns=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]).values

x

y=df.Species.values.reshape(-1,1)

y

"""### Importing necessary Libraries"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

"""## Train Test Split"""

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)

x_train.shape

y_train.shape

k=6
knclr=KNeighborsClassifier(k)

"""## Training the model using KNN (K Nearest Neighbor)"""

knclr

knclr.fit(x_train,y_train)

y_pred=knclr.predict(x_test)

metrics.accuracy_score(y_test,y_pred)*100

