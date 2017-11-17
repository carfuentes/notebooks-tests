
# coding: utf-8

# In[4]:


import numpy as np
import random as rand


# In[38]:


def NARMA_task(steps, data, init_len, train_len):
        Y=np.zeros(train_len)
        for t in range(init_len,train_len):
            Y[t]=0.3* Y[t-1] + 0.05*Y[t-1]*np.sum(Y[t-1:t-steps])+ 1.5*data[t-steps]*data[t-1]+0.1
                
        return Y


# In[39]:


data= [rand.uniform(0,0.5) for el in range(3000)]

NARMA_result= NARMA_task(10,data,200,2000)


# In[40]:


print(NARMA_result.shape)

