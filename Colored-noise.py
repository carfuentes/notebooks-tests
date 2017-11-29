
# coding: utf-8

# In[2]:


import numpy as np
from matplotlib.pyplot import * 
from math import sqrt
import numba


# In[3]:


@numba.jit(nopython=True)
def colored_noise_euler_integration(x_0, tau, c, dt=0.01, t_stop=60):
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


# In[5]:


# Specify parameters
x_0 = 0
tau = 1
c=1
dt=0.001
t_stop=10000
# Perform the solution
t, x = colored_noise_euler_integration(x_0, tau, c, dt, t_stop)
indexes=[int(t/dt) for t in range(0,t_stop)]
# Plot the result
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plot(t[:6000], x[:6000])
xlabel('time')
ylabel('x')
show()
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plot(t[indexes], x[indexes])
show()


# In[8]:


len(t[indexes])
t[indexes]


# In[55]:


sqrt(4)


# In[73]:


t


# In[91]:


int(10/0.001)
t[int(4/0.001)]


# In[87]:


np.where(t==1.00000000e+04)


# In[102]:


x=np.zeros((10,16))

x[0,:]=np.random.rand(16)
print(x)


# In[104]:


for i in range(0,10-1):
    x[i+1,:]=np.random.rand(16)


# In[105]:


x


# In[112]:


np.array([2,2,2])**2


# In[111]:


len(t[indexes])

