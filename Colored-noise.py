
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import * 
from math import sqrt


# In[ ]:


t= np.linspace(0,6,100)
D=1
lamda=1


# In[3]:


def colored_noise_euler_integration(x_0, lamda, D, n, dt=0.001, t_stop=6):
    """
    Use Euler integration to solve ODEs
    """    
    E=np.exp(-lamda * dt)
    
    # Time points
    t = np.linspace(0, t_stop, int(t_stop/dt))
    
    # Initialize output array
    x = x_0 * np.ones_like(t)
    
    # Do Euler stepping
    for i in range(i_time, len(t) - 1):
        h= sqrt( D * lamda * (1-E**2) ) * np.random.normal()
        x[i+1] = x[i] + dt * dx_dt(x[i - i_time], x[i], beta, tau, n)
        
    return t, x


# In[ ]:


def de_dt (e0,t, lamda,D):
    


# In[4]:


np.random.normal()


# In[5]:


np.exp(1*0.001)

