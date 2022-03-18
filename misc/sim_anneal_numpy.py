

import numpy as np
import matplotlib.pyplot as plt


#objective function
def h(x):
    if x<-1 or x>1:
        y = 0
    else:
        y = -((x+5)*(x+3)*(x+1)*(2-x))
    return y

hv = np.vectorize(h)
#turn the function into a vector so we can append it 
X = np.linspace(-5, 5, num=1000)

plt.plot(X, hv(X))


def SA(search_space, func, T):
    scale = np.sqrt(T)
    start = np.random.choice(search_space)
    x = start * 1
    cur = func(x)
    history = [x]
    for i in range(1000):
        prop = x + np.random.normal()*scale
        if prop > 1 or prop < 0 or np.log(np.random.rand()) * T > (func(prop) - cur):
            prop = X
        x = prop
        cur = func(x)
        T = 0.9 * T
        #decrease the temp by 10% at each iteration
        history.append(x)
        return x, history
    

x1, history = SA(X, h, T=4)
plt.plot(X, hv(X))
plt.scatter(x1, hv(x1), marker='x')
plt.plot(history, hv(history))
plt.show()