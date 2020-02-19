import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

data = pd.read_csv('D://TasteGraph/selfproject/Linear_regression/lsd.csv',low_memory =False)
#printing the columns in a data
#print (list(data))

#setting x and y values in a variable
X = data['Tissue Concentration'].values[:,np.newaxis]
y = data['Test Score'].values


#applying linear regression
model = LinearRegression()
model.fit(X, y)

#plotting  the model - dependent variable in the Y axis and independent variable in the X axis
plt.scatter(X, y,color='r')
plt.plot(X, model.predict(X),color='k')
plt.show()