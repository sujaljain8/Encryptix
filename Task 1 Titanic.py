#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv(r'C:\Users\Sujal Jain\Titanic.csv')


# In[4]:


df.info()


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()#or df.isna().sum()


# In[7]:


df.shape


# # Data Cleaning

# In[8]:


df.drop('PassengerId',inplace=True,axis=1)


# In[9]:


df['Cabin'].count()/df.shape[0]


# In[10]:


df.drop('Cabin',inplace=True,axis=1)#because there is more than 70% empty values


# In[11]:


df.info()


# In[12]:


df['Age'].count()/df.shape[0]


# In[13]:


df.dropna(subset=['Age','Embarked'],inplace=True)


# In[14]:


df.info()


# In[15]:


df['Title']=df['Name'].str.extract(r'([A-Za-z]+\.)',expand=False)
df.drop('Name',inplace=True,axis=1)


# In[16]:


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
df


# In[17]:


df['Survived'].value_counts()


# # EDA

# In[18]:


sns.countplot(data=df,x='Survived')
plt.ylabel('Number Of Survived')
plt.title('Survival Rate')
plt.grid()
plt.show()


# In[19]:


sns.countplot(x='Sex',data=df)
plt.ylabel('Number Of Gender')
plt.title('Gender Rate')
plt.grid()
plt.show()


# In[20]:


df['Sex'].value_counts()


# In[21]:


df['Pclass'].value_counts()


# In[22]:


sns.countplot(x='Pclass',data=df)
plt.ylabel('Number Of Classes')
plt.title('Classes')
plt.show()


# In[23]:


sns.histplot(df['Age'],bins=25)
plt.yticks(np.arange(0,100,10))
plt.xticks(np.arange(0,90,5))
plt.title('Age Distribution',fontsize=14)
plt.xlabel('No Of People')
plt.show()


# In[24]:


sns.countplot(x='SibSp',data=df)
plt.show()


# In[25]:


sns.countplot(x='Parch',data=df)


# In[26]:


print('Siblings:\n',df['SibSp'].value_counts())
print('Parents : \n',df['Parch'].value_counts())


# In[27]:


df.min()


# In[28]:


df.mean()


# In[29]:


df.max()


# In[30]:


df.mode()


# In[31]:


ind=df[df['Age']>30].index
df_1=df.drop(ind,axis=0)
sns.catplot(x='Survived',hue='Sex',kind='count',data=df_1)
plt.show()


# In[32]:


df.groupby('Survived')['Sex'].value_counts()


# In[33]:


sns.catplot(x='Survived',hue='Sex',col='Pclass',data=df,kind='count')
plt.yticks(np.arange(0,250,20))
plt.show()


# In[34]:


df.groupby(['Survived','Pclass'])['Sex'].value_counts()


# In[35]:


sns.catplot(x='Survived',hue='Sex',col='Embarked',data=df,kind='count')
plt.yticks(np.arange(0,330,30))
plt.show()


# In[36]:


df.groupby(['Survived','Embarked'])['Sex'].value_counts()


# In[37]:


sns.catplot(x='Survived',hue='Sex',col='Embarked',row='Pclass',data=df,kind='count')
plt.yticks(np.arange(0,190,20))
plt.tight_layout()
plt.show()


# In[38]:


df.groupby(['Survived','Embarked','Pclass'])['Sex'].value_counts()


# In[39]:


sns.heatmap(df.corr())


# In[40]:


sns.boxplot(df['Age'])
plt.show()


# In[41]:


def boundaries(data,col,dis):
    Q1=data[col].quantile(0.25)
    Q3=data[col].quantile(0.75)
    IQR=Q3-Q1
    low=Q1-(IQR*dis)
    upper=Q3+(IQR*dis)
    return low,upper


# In[42]:


lower,upper=boundaries(df,'Age',1.5)
print('Lower Range : ',lower,' Upper Range : ',upper)


# In[43]:


#Age_out=np.where(df['Age']>upper,True,(np.where(df['Age']<lower,True,False)))
not_out=(df['Age']<upper)&(df['Age']>lower)
df['Age'][~not_out].count()


# In[44]:


df=df[not_out]


# In[45]:


df['Agebin']=pd.cut(df['Age'],5,labels=['a','b','c','d','e'],include_lowest=True)
sns.countplot(x='Agebin',data=df)
plt.show()


# In[46]:


df


# # Applying ML

# In[47]:


df.drop(['Ticket','Agebin'],axis=1,inplace=True)


# In[48]:


from sklearn.preprocessing import LabelEncoder
lb_en=LabelEncoder()
df['Sex']=lb_en.fit_transform(df['Sex'])
df


# In[49]:


df['Embarked']=lb_en.fit_transform(df['Embarked'])
df['Title']=lb_en.fit_transform(df['Title'])
df


# In[88]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score,confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier


# In[51]:


x=df.drop('Survived',axis=1)
y=df['Survived']


# In[52]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.3)


# In[53]:


Li_Re=LinearRegression()


# In[54]:


Li_Re.fit(x_train,y_train)


# In[55]:


y_test_pre=Li_Re.predict(x_test)


# In[56]:


y_train_pre=Li_Re.predict(x_train)


# In[57]:


r2_score(y_test,y_test_pre)


# In[58]:


r2_score(y_train,y_train_pre)


# In[59]:


#since it is classification problem not regression so


# In[60]:


#lg_Re=LogisticRegression(max_iter=500)
best_params = {
    'C': 1,
    'max_iter': 100,
    'penalty': 'l1',
    'solver': 'liblinear'
}
lg_Re = LogisticRegression(**best_params)


# In[61]:


lg_Re.fit(x_train,y_train)


# In[62]:


y_test_pre=lg_Re.predict(x_test)


# In[63]:


y_train_pre=lg_Re.predict(x_train)


# In[64]:


print(classification_report(y_test,y_test_pre))


# In[65]:


print(classification_report(y_train,y_train_pre))


# In[66]:


print('Accuracy : ',lg_Re.score(x_test,y_test)*100)


# In[67]:


kfd=KFold(n_splits=10,random_state=100,shuffle=True)


# In[68]:


res=cross_val_score(lg_Re,x,y,cv=kfd,n_jobs=-1)
print('Accuracy : ',res.mean()*100)


# In[69]:


rkfd=RepeatedKFold(n_splits=10,random_state=100,n_repeats=2)
res1=cross_val_score(lg_Re,x,y,cv=rkfd,n_jobs=-1)
print('Accuracy : ',res1.mean()*100)


# In[70]:


skflod=StratifiedKFold(n_splits=10,random_state=100,shuffle=True)
res2=cross_val_score(lg_Re,x,y,cv=skflod,n_jobs=-1)
print('Accuracy : ',res2.mean()*100)


# In[71]:


from sklearn.model_selection import GridSearchCV

# Define the hyperparameters and their possible values for tuning
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga'],  # Include only solvers compatible with l1 penalty
    'max_iter': [100, 200, 300,500]
}


# Create a GridSearchCV instance
grid_search = GridSearchCV(lg_Re, param_grid, cv=5, n_jobs=-1)

# Fit the grid search to your data
grid_search.fit(x_train, y_train)

# Get the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


# In[72]:


print(best_model,best_params)


# In[73]:


from sklearn.tree import DecisionTreeClassifier
#DTC=DecisionTreeClassifier()
DTC= DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features='log2',
                       min_samples_leaf=2, splitter='random')


# In[74]:


DTC.fit(x_train,y_train)


# In[75]:


y_pre_test=DTC.predict(x_test)


# In[76]:


y_pre_train=DTC.predict(x_train)


# In[77]:


print(classification_report(y_test,y_pre_test))


# In[78]:


print(classification_report(y_train,y_pre_train))


# In[79]:


print(DTC.score(x_test,y_test)*100)


# In[80]:


kflod=KFold(n_splits=220,random_state=100,shuffle=True)
D_res=cross_val_score(DTC,x,y,cv=kflod,n_jobs=-1)
print('Accuracy : ',D_res.mean()*100)


# In[81]:


refold=RepeatedKFold(n_splits=220,random_state=100,n_repeats=3)
D_res1=cross_val_score(DTC,x,y,cv=refold,n_jobs=-1)
print('Accuracy : ',D_res1.mean()*100)


# In[82]:


skfoldd=StratifiedKFold(n_splits=220,random_state=100,shuffle=True)
D_res2=cross_val_score(DTC,x,y,cv=skfoldd,n_jobs=-1)
print('Accuracy : ',D_res2.mean()*100)


# In[105]:


param_grid1 = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2'],
    'splitter': ['best', 'random'],
    'class_weight': [None, 'balanced']}
    
gir_sear=GridSearchCV(DTC,param_grid1,cv=5,n_jobs=-1)
gir_sear.fit(x_train,y_train)
print(gir_sear.best_params_,gir_sear.best_estimator_,gir_sear.best_score_)


# In[107]:


#RFC=RandomForestClassifier()
RFC=RandomForestClassifier(bootstrap= False, max_depth=10, max_features= 'sqrt', min_samples_leaf= 4, min_samples_split= 10, n_estimators= 50,random_state=42)


# In[108]:


RFC.fit(x_train,y_train)
y_rfc_pre_test=RFC.predict(x_test)
y_rfc_pre_train=RFC.predict(x_train)


# In[109]:


print(classification_report(y_train,y_rfc_pre_train))
print(classification_report(y_test,y_rfc_pre_test))
print(confusion_matrix(y_test,y_rfc_pre_test))


# In[110]:


print(RFC.score(x_test,y_test))


# In[111]:


kfd_rfc=KFold(n_splits=10,shuffle=True,random_state=100)
res_kfd=cross_val_score(RFC,x,y,cv=kfd_rfc,n_jobs=-1)
print('Accuracy : ',res_kfd.mean()*100)


# In[112]:


rkfd_rfc=RepeatedKFold(n_splits=10,n_repeats=3,random_state=100)
res_rkfd=cross_val_score(RFC,x,y,cv=rkfd_rfc,n_jobs=-1)
print('Accuracy : ',res_rkfd.mean()*100)


# In[113]:


sk_res=StratifiedKFold(n_splits=10,shuffle=True,random_state=100)
res_sk=cross_val_score(RFC,x,y,cv=sk_res,n_jobs=-1)
print('Accuracy : ',res_sk.mean()*100)


# In[104]:


param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt'],
    'bootstrap': [True, False]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=RFC, param_grid=param_grid, 
                           scoring='accuracy', cv=5, n_jobs=-1)

# Fit the GridSearchCV object to your data
grid_search.fit(x_train, y_train)  # Replace X_train and y_train with your data

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

