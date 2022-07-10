#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



# In[13]:


import matplotlib as pyplot


# In[14]:


data = pd.read_csv("diabetes.csv")
data.head()


# In[15]:


# outcome= 1 diabet
# outcome= 0 saglikli


# In[16]:




seker_hastalari = data[data.Outcome == 1]
saglikli_insanlar = data[data.Outcome == 0]


# In[17]:





y = data.Outcome.values
x_ham_veri = data.drop(["Outcome"],axis=1)   # axis=1 sütun, axis=0 satir
# Sadece bagimsiz degisken birakiyoruz cunku knn siniflandirma  yapacak.


x = (x_ham_veri - np.min(x_ham_veri))/(np.max(x_ham_veri)-np.min(x_ham_veri))  # normalizasyon



# In[18]:


print("Normalization öncesi veriler:\n")
print(x_ham_veri.head())

print("\nNormalization sonrası veriler:\n")
print(x.head())
    


# In[22]:



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=1)

# knn modeli
knn = KNeighborsClassifier(n_neighbors = 10) # n_neighbors = k
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("K=3 için Test verilerimizin doğrulama testi sonucu: ", knn.score(x_test, y_test))


# In[23]:



from sklearn.preprocessing import MinMaxScaler


sc = MinMaxScaler()  #hizli normalization
sc.fit_transform(x_ham_veri)

new_prediction = knn.predict(sc.transform(np.array([[6,148,72,35,0,33.6,0.627,50]])))
new_prediction[0]


# In[ ]:




