""" REGRESION LINEAL """

from statistics import mean  #Para calcular la media de todos los datos
import numpy as np  #Permite usar arrays en python
import matplotlib.pyplot as plt  #Permite hacer graficas
from matplotlib import style
import random

style.use('fivethirtyeight')  #este es el estilo que elegimos
#style.use('ggplot')  #este es el estilo que elegimos



#xs = np.array([1,2,3,4,5,6], dtype=float)
#ys = np.array([5,6,4,7,6,8], dtype=float)

def create_database(hm, variance, step=1, correlation=False):
  val = 1
  ys = []
  for i in range(hm):
    y = val + random.randrange(-variance,variance)
    ys.append(y)
    if correlation and correlation == 'pos':
      val += step
    elif correlation and correlation == 'neg':
      val -+ step
  xs = [i for i in range(hm)]
    
  return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

#plt.scatter(xs,ys) #genera los puntos
#plt.show #Muestra la gráfica

def best_fit_slope_and_b(xs, ys):
  
  numerator = (np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)
  denominator = (np.mean(xs)**2) - (np.mean(xs**2))

  m = numerator/denominator

  b = np.mean(ys) - (m*np.mean(xs))

  return b, m

def squared_error(y_original, y_prediccion):
  return sum( ( y_original - y_prediccion )**2 )

def coeficient_of_determination(y_original, y_prediccion):

  y_mean_line = [mean(y_original) for y in y_original]
  squared_error_predict = squared_error(y_original, y_prediccion)
  squared_error_mean = squared_error(y_original, y_mean_line)

  return (1 - ( squared_error_predict / squared_error_mean ))
  
xs, ys = create_database(40,5,2,correlation='pos')

b ,m = best_fit_slope_and_b(xs,ys)

regression_line = [(b + m*x) for x in xs]

rcuadrado = coeficient_of_determination(ys, regression_line)

print(rcuadrado)

print(regression_line)
print(b,m)

plt.scatter(xs,ys) #genera los puntos
plt.plot(xs,regression_line)
plt.show #Muestra la gráfica

