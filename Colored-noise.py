
# coding: utf-8

# In[185]:


import numpy as np
from matplotlib.pyplot import * 
from math import sqrt, pi,cos,log
import numba
import scipy.signal
from statsmodels.graphics.tsaplots import plot_acf


# In[2]:


@numba.jit(nopython=True)
def colored_noise_euler_integration(x_0, tau, c, D, dt=0.001, t_stop=101):
    """
    Use Euler integration to solve ODEs
    """    
    mu=np.exp(-dt/tau)
    sigma= sqrt( ((c * tau)/2) * (1-mu**2) )
    
    # Time points
    t = np.linspace(0, t_stop, int(t_stop/dt))
    
    # Initialize output array
    x = x_0 * np.ones_like(t)
    
    for i in range(0, len(t) - 1):
        x[i+1] = x[i]* mu + sigma * np.random.normal()
        
    return t, x


# In[3]:


#@numba.jit(nopython=True)
def colored_noise_euler_integration_fox(x_0, lamda, D, dt=0.001, t_stop=101):
    """
    Use Euler integration to solve ODEs
    """    
    E=np.exp(-lamda * dt)
    
    # Time points
    t = np.linspace(0, t_stop, int(t_stop/dt))
    
    # Initialize output array
    x = x_0 * np.ones_like(t)
    
    for i in range(0, len(t) - 1):
        a=np.random.uniform()
        b=np.random.uniform()
        x[i+1] = x[i]* E + sqrt(-2*D*lamda*(1-E**2)*np.log(a))*cos(2*pi*b)
        
    return t, x


# In[196]:


# Specify parameters
x_0 = 1
tau =10
c=1
dt=0.001
D=10
lamda=0.003
t_stop=100


# In[173]:


##GILLESPIE


# In[174]:


print(np.exp(1))


# In[197]:


# Perform the solution
t, x = colored_noise_euler_integration(x_0, tau, c, D, dt, t_stop)

# Plot the result
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plot(t, x)
xlabel('time')
ylabel('x')
show()


# In[176]:


##FOX


# In[177]:


# Perform the solution
t, x = colored_noise_euler_integration_fox(x_0, lamda,D, dt, t_stop)

# Plot the result
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plot(t, x)
xlabel('time')
ylabel('x')
show()


# In[ ]:


## AUTOCORRELATION


# In[84]:


def autocorrelation(x):
    result = np.correlate(x, x, mode='full')
    result=result[int(result.size/2):]
    return result/result[0]


# In[198]:


#1
autocorr=autocorrelation(x)
plot(autocorr)
show()


# In[135]:


#2
y = x - np.mean(x)
norm = np.sum(y ** 2)
correlated = np.correlate(y, y, mode='full')/norm
plot(correlated)
show()


# In[134]:


#3
plot_acf(x)
show()


# In[44]:


#4
iact=[]
timeseries = x
mean = np.mean(timeseries)
timeseries -= np.mean(timeseries)
autocorr_f = np.correlate(timeseries, timeseries, mode='full')
temp = autocorr_f[int(autocorr_f.size/2):]/autocorr_f[int(autocorr_f.size/2)]
iact.append(sum(autocorr_f[int(autocorr_f.size/2):]/autocorr_f[int(autocorr_f.size/2)]))

