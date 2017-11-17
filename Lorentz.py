
# coding: utf-8

# In[2]:


import scipy.signal
import numpy as np
import matplotlib.pyplot as plt


# In[22]:


#Parámetros
sigma=10.0
beta=8/3
rho=28

#INITIAL CONDITION
x0=4.0
y0=1.0
z0=1.0

x0_y0_z0=np.array([x0,y0,z0])

#t
t=np.linspace(0,200,int(250*2000))
print(t)


# In[23]:


def dx_y_z(x_y_z,t,sigma,rho,beta):
    x,y,z=x_y_z
    dx_dt=sigma * (y-x)
    dy_dt=x * (rho-z) -y
    dz_dt=x *y -beta*z
    return np.array([dx_dt,dy_dt,dz_dt])


# In[24]:


x_y_z=scipy.integrate.odeint(dx_y_z,x0_y0_z0, t,args=(sigma,rho,beta))
x=x_y_z[:,0]


# In[25]:


plt.plot(t,x)
plt.show()


# In[26]:


plt.plot(x_y_z[:,0], x_y_z[:,1])
plt.show()

