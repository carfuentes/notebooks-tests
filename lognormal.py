
# coding: utf-8

# In[3]:


from statistics import mean
import numpy as np


# In[21]:


def random_sampling_normal_from_range(list_range,size):
    m= mean(list_range)
    s= abs((list_range[1] - m)/3)
    return np.random.normal(m,s,size)
    
    


# In[20]:


random_sampling_normal_from_range([0.02,0.2],1)

