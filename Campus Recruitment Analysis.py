#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

# In[1]:


import numpy as np
import pandas as pd


# ### Loading the dataset

# In[2]:


df = pd.read_csv('Placement_Data_Full_Class.csv')
df.head()


# In[3]:


df.shape


# ### Checking the data types of features

# In[4]:


df.info()


# ### Checking various statistical values of all the numerical features

# In[5]:


df.describe().T


# ### Checking for missing values

# In[6]:


df.isna().sum()


# ### We can see that 'salary' feature has missing values, this maybe because the candidate did not get placed, so the salary will be 0, so we replace NaN values with 0

# In[7]:


df['salary'] = df['salary'].fillna(0)


# In[8]:


df.isna().sum()


# In[9]:


df.head()


# ### Plotting these data

# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Plotting the count of male and female candidates

# In[11]:


sns.countplot(x = 'gender', data = df,palette = 'magma')
plt.title('Male and Female count')
plt.xlabel('Gender')
plt.ylabel('Number of students')


# In[12]:


sns.countplot(x = 'status', data = df,hue = 'gender',palette = 'dark')


# In[13]:


# senior secondary percentage
sns.violinplot(x = 'status',  y ='ssc_p',data = df, palette = 'summer') 


# In[14]:


#senior secondary board
sns.countplot(x = 'ssc_b', data = df, hue ='status', palette = 'viridis') 
plt.xlabel('Senior Secondary Board')
plt.ylabel('Number of students')


# In[15]:


#higher secondary percentage
sns.violinplot(x = 'status',y = 'hsc_p' ,data = df, palette = 'PuBu_r') 


# In[16]:


#higher secondary board
sns.countplot(x = 'hsc_b', data = df, hue ='status', palette = 'prism_r') 
plt.xlabel('Higher Secondary Board')
plt.ylabel('Number of students')


# In[17]:


#higher secondary board
sns.countplot(x = 'hsc_s', data = df, hue ='status') 
plt.xlabel('Specialization')
plt.ylabel('Number of students')


# In[18]:


#higher secondary percentage
sns.violinplot(x = 'status',y = 'degree_p' ,data = df, palette = 'RdBu') 


# In[19]:


#degree type(degree_t)
sns.countplot(x = 'degree_t', data = df, hue = 'status', palette = 'twilight')
plt.title('Graduation Degree vs  Placement')
plt.xlabel('Graduation Degree')
plt.ylabel('Number of students')
plt.show()


# In[20]:


#work experience
plt.figure(figsize= (10,6))
sns.boxplot(x = 'workex', y = 'salary',data = df)


# In[21]:


#mba specialization
sns.countplot(x = 'specialisation', data = df, hue = 'status')
plt.title('Specialization vs  Placement')
plt.xlabel('Specialization')
plt.ylabel('Number of students')
plt.show()


# In[22]:


#mba percentage
sns.violinplot(x = 'status',y = 'mba_p' ,data = df, palette = 'twilight') 


# ### Data Preprocessing

# In[23]:


df.head()


# In[24]:


df= df.drop(['sl_no','salary'], axis = 1)


# In[25]:


df.head()


# In[26]:


df.info()


# ### Converting categorical variables into numerical variables

# In[27]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['ssc_b'] = le.fit_transform(df['ssc_b'])
df['hsc_b'] = le.fit_transform(df['hsc_b'])
df['hsc_s'] = le.fit_transform(df['hsc_s'])
df['degree_t'] = le.fit_transform(df['degree_t'])
df['workex'] = le.fit_transform(df['workex'])
df['specialisation'] = le.fit_transform(df['specialisation'])
df['status'] = le.fit_transform(df['status'])


# In[28]:


df.head()


# In[29]:


plt.figure(figsize=(15,10))
corr = df.corr()
sns.heatmap(corr, annot = True)


# In[30]:


X = df.drop('status', axis = 1) #independent features
y = df['status'] #dependent feature


# In[31]:


X


# In[32]:


y


# In[33]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)


# In[34]:


X


# ### Splitting data into training and test sets

# In[35]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=121)


# In[36]:


X_train


# In[37]:


X_train.shape


# In[38]:


y_train.shape


# In[39]:


from sklearn.linear_model import LogisticRegression


# In[40]:


model = LogisticRegression()
model.fit(X_train,y_train)
pred= model.predict(X_test)


# In[41]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[42]:


cm = confusion_matrix(y_test,pred)
sns.heatmap(cm,annot = True)


# In[43]:


print(classification_report(y_test,pred))


# In[44]:


print("The accuracy of the model is: ",accuracy_score(y_test,pred))


# In[ ]:




