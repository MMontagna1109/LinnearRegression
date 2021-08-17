# LinnearRegression
Curso SentDex


#!pip install quandl
#!pip install sklearn
#!pip install pandas


import pandas as pd
import quandl  #Es la base de datos de las acciones
import math  #Permite hacer operaciones matematicas
import datetime  #Permite trabajar con fecha y horas
import numpy as np  #Permite utilizar arrays, python no deja usar arrays sin esta libreria
from sklearn import preprocessing, svm, model_selection
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style #Esto es para elegir estilos
import pickle 

style.use('ggplot')  #este es el estilo que elegimos

df = quandl.get("WIKI/GOOGL",authtoken='RmA9UyHag92sdMTzyurh')

#print(df.head())

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df['HL_PCT'] = (df['Adj. High']-df['Adj. Low'])/df['Adj. Low']

df['OC_PCT'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']

#print(df['HL_PCT'].head())

#print(df['OC_PCT'].head())

df = df[['Adj. Close','HL_PCT','OC_PCT','Adj. Volume']]
#print(df.head())

forecast_col = 'Adj. Close' #Si quiero cambiar lo que trato de predecir, tnego que cambiar solamente esta linea
df.fillna(value=-99999,inplace=True)  #Esto es para cambiar todos los NaN(not a number) que haya en los datos con el valor -9999

forecast_out = int(math.ceil(0.009*len(df))) #Forecast out = pronosticar --> con el o.1 cambio que tanto quiero pronosticar. El int no hace falta, math.ceil ya devuelve un entero
#print(forecast_out) #Para ver cuantos dias en adelante predice el modelo

df['Label'] = df[forecast_col].shift(-forecast_out)

#print(df.head())

#""" En el eje X pongo las features, y en el Y las etiquetas """
X = np.array(df.drop(['Label'], 1))  # Quito de la base de datos la columna 'Label'. .drop genera una nueva base de datos
X = preprocessing.scale(X)  #Normaliza los datos, pero agrega tiempo de computo.
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['Label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)  #Separa y aleatoriza los datos, y pone un 20% para testear

clf = LinearRegression()  #Libreria de TensorFlow
clf.fit(X_train, y_train) #Utilizo la separacion de train para entrenar (%80)

accuracy = clf.score(X_test, y_test) #Chequeas con el 20% restante de la base de datos que tan bien se comporta el modelo

#print(accuracy) #Esta exactitud se mide en la diferencia de cuadrados

forecast_set = clf.predict(X_lately)
#print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan  #lleno la columna Forecast con valores NaN

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:  #Agrego la fecha como indice a cada prediccion, y relleno con NaN todas las columnas
  next_date = datetime.datetime.fromtimestamp(next_unix)
  next_unix += one_day
  df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.head)

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
