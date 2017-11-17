
# coding: utf-8

# In[2]:


import networkx as nx
import numpy as np
import random as rand
import scipy
from matplotlib.pyplot import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
import os
from math import sqrt
from math import sin
import json
from statistics import mean
import entropy_estimators as ee
import scipy.spatial as ss
from scipy.special import digamma,gamma
from math import log,pi
import numpy.random as nr



# In[3]:


def random_sampling_normal_from_range(list_range,size):
    m= mean(list_range)
    s= abs((list_range[1] - m)/3)
    return np.random.normal(m,s,size)


# In[4]:


#networkx 1.10
def get_cyclic_net_v1(filename):
    G=nx.read_edgelist(filename, comments='#', delimiter="\t", nodetype =str,  data=(('mode',str),), create_using=nx.DiGraph())
    G.remove_nodes_from(["Source", "Target"])
    selfloops=G.selfloop_edges()
    G.remove_edges_from(G.selfloop_edges())
    
    
    while 0 in G.in_degree().values() or 0 in G.out_degree().values():
        nodes_to_remove=[node for node in G if G.in_degree(node) == 0 or G.out_degree(node) == 0]
        G.remove_nodes_from(nodes_to_remove)
        
    selfloops_in_reservoir=[edge for edge in selfloops if edge[0] in G.nodes()]
    G.add_edges_from(selfloops_in_reservoir)
    return G
    


# In[5]:


#networkx 2.1
def get_cyclic_net(filename):
    G=nx.read_edgelist(filename, comments='#', delimiter="\t", nodetype =str,  data=(('mode',str),), create_using=nx.DiGraph())
    G.remove_nodes_from(["Source", "Target"])
    selfloops=G.selfloop_edges()
    G.remove_edges_from(G.selfloop_edges())

    while 0 in [d[1] for d in G.in_degree()] or 0 in [d[1] for d in G.out_degree()]:
        nodes_to_remove=[node for node in G if G.in_degree(node) == 0 or G.out_degree(node) == 0]
        G.remove_nodes_from(nodes_to_remove)
        
    selfloops_in_reservoir=[edge for edge in selfloops if edge[0] in G.nodes()]
    G.add_edges_from(selfloops_in_reservoir)
    return G


# In[6]:


class ESN(object):
    def __init__(self, filename, in_size, out_size, spectral_radius):
        self.res_size= self.build_adj_weighted_matrix(filename).shape[0]
        self.in_size=in_size
        self.out_size=out_size
        self.spectral_radius= spectral_radius
        self.W0=self.build_adj_weighted_matrix(filename)
        self.W=None
        self.Win=None
        self.Wout=None
        self.X=None
        self.Y=None
        self.x=np.zeros((self.res_size,1))
        self.x0=np.insert(np.random.rand(self.res_size)*10,0,[1.0,1.0,1.0])
        self.decay=random_sampling_normal_from_range([1/15,1/60],(self.res_size,1))
        self.u=None
        self.x_act=None

    
    def build_adj_weighted_matrix(self, filename):
        #NETWORK v2.0
        net=get_cyclic_net(filename)
        for edge in net.edges(data="mode", default=0):
            source,target,mode=edge
            if mode== "+":
                net[source][target]["weight"]= rand.uniform(0,1)
            elif mode== "-":
                net[source][target]["weight"]= rand.uniform(0,-1)
            elif mode== 0:
                net[source][target]["weight"]= rand.uniform(-1,1)
        return nx.to_numpy_matrix(net)
    
    def initialize(self): 
        np.random.seed(42)
        self.Win=np.random.choice([-0.05,0.05], size=(self.res_size,1+self.in_size))
        self.W0 = np.squeeze(np.asarray(self.W0))
        rhoW0 = max(abs(scipy.linalg.eig(self.W0)[0]))
        self.W= (self.spectral_radius/rhoW0)*self.W0
        

    def collect_states(self, data, init_len, train_len, a=0.3):
        self.X=np.zeros((self.res_size+self.in_size+1, train_len-init_len))
        for t in range(train_len):
            u = data[t]
            self.x = (1-a)*self.x + a*np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ) ) 
            if t >= init_len:
                self.X[:,t-init_len]= np.vstack((1,u,self.x))[:,0]
        
        return self.X
    
    
    def collect_states_derivative(self, a,b,c, init_len, train_len, test_len):
        self.X=np.zeros((self.res_size+self.in_size+1, train_len-init_len))
        t=np.arange(train_len+test_len)
        uyz_x=scipy.integrate.odeint(self.dx_dt,self.x0,t,args=(a,b,c,self.decay))
        self.u=uyz_x[:,0]
        self.x_act=uyz_x[:,3:]
        for t in range(init_len,train_len):
            x_concat=self.x_act[t,:].reshape(self.x_act[t,:].shape[0],1)
            u_concat=self.u[t]
            self.X[:,t-init_len]= np.vstack((1,u_concat,x_concat))[:,0]
               
        return self.X
    
    def dx_dt(self, uyz_x,t,a,b,c,decay):
        u=uyz_x[0]
        y=uyz_x[1]
        z=uyz_x[2]
        x=np.array(uyz_x[3:]).reshape(self.res_size,1)
       
        du_dt=-z-y
        dy_dt=u+a*y
        dz_dt=b+z*(u-c)
        dx_dt= np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, x ) ) - (decay * x)
        
        return np.insert(dx_dt,0,[du_dt,dy_dt,dz_dt])
        
    def calculate_weights(self, data, init_len, train_len,beta=1e-8 ):
        Y=data[None,init_len+1:train_len+1]
        X_T=self.X.T
        self.Wout= np.dot ( np.dot(Y, X_T), np.linalg.inv(np.dot(self.X,X_T) + beta * np.eye(self.res_size+self.in_size+1)))
        return self.Wout
    
    def calculate_weights_derivative(self,init_len, train_len, n, beta=1e-8 ):
        Y=np.array([self.u[init_len+1-n:train_len+1-n]])
        X_T=self.X.T
        self.Wout= np.dot ( np.dot(Y, X_T), np.linalg.inv(np.dot(self.X,X_T) + beta * np.eye(self.res_size+self.in_size+1))) #w= y*x_t*(x*x_t + beta*I)^-1
        return self.Wout
    
    def run_generative(self, data, test_len, train_len,a=0.3):
        self.Y = np.zeros((self.out_size,test_len))
        u = data[train_len]
        for t in range(test_len):
            self.x = (1-a)*self.x + a*np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ) ) 
            y = np.dot( self.Wout, np.vstack((1,u,self.x)) )
            self.Y[:,t] = y
            u = data[trainLen+t+1]
            #u =y
    
    def run_predictive_derivative(self, a,b,c, test_len, train_len):
        self.Y = np.zeros((self.out_size,test_len))
        
        for t in range(train_len,train_len+test_len):
            x_concat=self.x_act[t,:].reshape(self.x_act[t,:].shape[0],1)
            u_concat=self.u[t]
            y = np.dot( self.Wout, np.vstack((1,u_concat,x_concat)) )
            self.Y[:,t-train_len] = y
           
        
        return self.Y


# In[7]:


##################################################################################


# In[8]:


#                                   FUNCTIONS                                    #


# In[9]:


#NARMA 
def NARMA_task(steps, data, init_len, train_len):
        Y=np.zeros(train_len)
        for t in range(init_len,train_len):
            Y[t]=0.3* Y[t-1] + 0.05*Y[t-1]*np.sum(Y[t-1:t-steps])+ 1.5*data[t-steps]*data[t-1]+0.1
                
        return Y


# In[10]:


def testing_gene_net(directory,input_data,data):
    csv_files= [file for file in os.listdir(directory) if file.startswith("network_edge_list")]
    print(csv_files)
    MI_by_file={}
    for file in csv_files:
        filename=file[file.index("list")+5:file.index(".csv")]
        net=ESN(os.path.join(directory, file),1,1,0.95)
        net.initialize()
        net.collect_states(input_data,initLen,trainLen)
        net.calculate_weights(input_data,initLen,trainLen)
        net.run_generative(input_data,testLen,trainLen)
        MI_by_file[filename]=memory_capacity_n(net.Y, data,100)
        nrmse= sqrt(mean_squared_error(data[trainLen+1:trainLen+errorLen+1],net.Y[0,0:errorLen])/np.std(net.Y[0,0:errorLen]))
        print(net.res_size, 'NRMSE = ' + str( nrmse ))
        print(memory_capacity_n(net.Y, data,20))
        
    return MI_by_file
        #plot( data[trainLen+1:trainLen+testLen+1], 'g' )
        #plot( net.Y.T, 'b' )
        #title('Target and generated signals $y(n)$ starting at $n=0$')
        #legend(['Target signal', 'Free-running predicted signal'])
        #show()
      


# In[23]:


def testing_gene_net_derivative(directory,a,b,c,n):
    csv_files= [file for file in os.listdir(directory) if file.startswith("network_edge_list")]
    Y_by_file={}
    X_by_file={}
    MI_by_file={}
    NRMSE_by_file={}
    for file in csv_files:
        print(file)
        filename=file[file.index("list")+5:file.index(".csv")]
        net=ESN(os.path.join(directory, file),1,1,0.95)
        net.initialize()
        net.collect_states_derivative(a,b,c,initLen,trainLen,testLen)
        net.calculate_weights_derivative(initLen,trainLen,n)
        net.run_predictive_derivative(a,b,c,testLen,trainLen)
        X_by_file[filename]=net.u
        Y_by_file[filename]=net.Y
        NRMSE_by_file[filename]=nrmse_n(net.Y,net.u)
        MI_by_file[filename]=memory_capacity_n(net.Y, net.u,n)
        print(nrmse(net.Y[0,0:errorLen],net.u[trainLen+1:trainLen+errorLen+1]))
        print(net.res_size, " FINISHED")
        
        #figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
        #plot( net.u[trainLen+1:trainLen+testLen+1], 'g' )
        #plot( net.Y.T, 'b' )
        #xlabel("time")
        #ylabel("signal")
        #title('Target and generated signals $y(n)$ starting at $n=0$ until $n=%s$' %testLen)
        #legend(['Target signal', 'Free-running predicted signal'])
        #savefig("plots-input-vs-output/%s_testLen" %filename)
        #show()
        
        #figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
        #plot( net.u[trainLen+1:trainLen+50+1], 'g' )
        #plot( net.Y.T[0:50], 'b' )
        #xlabel("time")
        #ylabel("signal")
        #title('Target and generated signals $y(n)$ starting at $n=0$ until $n=50$')
        #legend(['Target signal', 'Free-running predicted signal'])
        #savefig("plots-input-vs-output/%s_50" %filename)
        #show()
       
  
    return MI_by_file, X_by_file, Y_by_file, NRMSE_by_file
  


# In[12]:


def testing_gene_net_file(directory,file):
    print(file)
    filename=file[file.index("list")+5:file.index(".csv")]
    net=ESN(os.path.join(directory, file),1,1,0.95)
    net.initialize()
    net.collect_states(data,initLen,trainLen)
    net.calculate_weights(data,initLen,trainLen)
    net.run_generative(data,testLen,trainLen)
    nrmse= sqrt(mean_squared_error(data[trainLen+1:trainLen+errorLen+1],net.Y[0,0:errorLen])/np.std(net.Y[0,0:errorLen]))
    print(net.res_size, 'NRMSE = ' + str( nrmse ))
    return memory_capacity_n(net.Y, data,100)
    


# In[13]:


def testing_gene_net_derivative_file(directory,file):
    print(file)
    filename=file[file.index("list")+5:file.index(".csv")]
    net=ESN(os.path.join(directory, file),1,1,0.95)
    net.initialize()
    net.collect_states_derivative(a,b,c,initLen,trainLen,testLen)
    net.calculate_weights_derivative(initLen,trainLen)
    net.run_predictive_derivative(a,b,c,testLen,trainLen)
    #nrmse= sqrt(mean_squared_error(net.u[trainLen+1:trainLen+errorLen+1],net.Y[0,0:errorLen])/np.std(net.Y[0,0:errorLen]))
    #print(net.res_size, 'NRMSE = ' + str( nrmse ))
    #plot(np.arange(trainLen+testLen+1),net.u)
    #show()
    return memory_capacity_n(net.Y,net.u,100)


# In[14]:


def entropy(x,k=3,base=2):
    """ The classic K-L k-nearest neighbor continuous entropy estimator
         x should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert k <= len(x)-1 #"Set k smaller than num. samples - 1"
    d = len(x[0])
    N = len(x)
    intens = 1e-10 #small noise to break degeneracy, see doc.
    x = [list(p + intens*nr.rand(len(x[0]))) for p in x]
    tree = ss.cKDTree(x)
    nn = [tree.query(point,k+1,p=float('inf'))[0][k] for point in x]
    const = digamma(N)-digamma(k) + d*log(2)
    return (const + d*np.mean(list(map(log,nn)),dtype=np.float64))/log(base)


# In[15]:


def entropy_binning(c_xy):
    c_normalized = c_xy/np.sum(c_xy)
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    h = -sum(c_normalized * np.log(c_normalized))  
    return h


# In[16]:


def calc_MI_binning(x, y):
    bins=sqrt(x.shape[0]/5)
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    #mi = mutual_info_score(x,y)
    return mi/entropy_binning(c_xy)


# In[17]:


def calc_MI_npeet(x,y):
    return ee.mi(x.reshape((x.shape[0],1)), y.reshape((y.shape[0],1)), base=2)/entropy(x.reshape((x.shape[0],1)))


# In[18]:


def memory_capacity_n(Yt, Xt,n):
    MI_i={}
    for i in range(200+1):
        MI_i[i]=calc_MI_npeet(Yt[0,300:900],Xt[300-i:900-i]) 
    
    return MI_i


# In[19]:


def nrmse_n(Yt, Xt):
    NRMSE_i={}
    for i in range(51):
        NRMSE_i[i]=nrmse(Yt[0,200:1000],Xt[200-i:1000-i]) 
    
    return NRMSE_i


# In[20]:


def nrmse(y,x):
    return sqrt(mean_squared_error(x,y))/np.std(x)


# In[48]:


def plot_MI_i(key,mi_dict,n):
    x=[]
    y=[]
    for i,MI in mi_dict.items():
        x.append(i)
        y.append(MI)
    plot(x,y,label=key)
    plot(n,mi_dict[n],marker='o')


# In[20]:


def plot_MI_by_file(MI_by_file):
    for key in MI_by_file.keys():
        plot_MI_i(key, MI_by_file[key])
    legend(loc='upper left')
    show()
    


# In[ ]:


##################################################################################


# In[ ]:


#                                   TESTEOS                                      #


# In[ ]:


# TESTEO get_cyclic_net
G=get_cyclic_net_v1(os.path.join("Dataset1/", "network_edge_list_modENCODE.csv"))
len(G.nodes())


# In[ ]:


#TESTEO adjacency matrix
net=ESN(os.path.join("Dataset1/", "network_edge_list_DBTBS.csv"),1,1,0.95)
net.W0


# In[ ]:


#TESTEO initialize
net.initialize()
print(net.W.shape)
print(max(abs(scipy.linalg.eig(net.W)[0])))


# In[ ]:


#TESTEO collect states
net.collect_states_derivative(a,b,c, initLen, trainLen, testLen, n=2)
net.X.shape
net.X[:,7]


# In[ ]:


##################################################################################


# In[ ]:


#                             RESULTS                                            #


# In[21]:


# TRAINING AND TEST LENGHT
errorLen = 500
trainLen=9000
testLen=1000
initLen=200

#Files
csv_files=['network_edge_list_ENCODE.csv', 'network_edge_list_modENCODE.csv', 'network_edge_list_YEASTRACT.csv', 'network_edge_list_EcoCyc.csv', 'network_edge_list_DBTBS.csv']

#parameters ROSSLER
a=0.1
b=0.1
c=14


# In[ ]:


#MACKEY GLASS


# In[ ]:


print("MACKEY GLASS")
data = np.loadtxt('MackeyGlass_t17.txt')
MI=testing_gene_net("Dataset1/",data,data)


# In[ ]:


# NARMA TASK


# In[ ]:


u_narma= [rand.uniform(0,0.5) for el in range(10501)]
NARMA_result= NARMA_task(10,u_narma,initLen,len(data))
NARMA_result[2000:4000]


# In[ ]:


print("NARMA")
testing_gene_net("Dataset1/",NARMA_result)


# In[ ]:


# ROSSLER


# In[ ]:


print("With derivatives")
MI_by_file, X_by_file, Y_by_file, NRMSE_by_file=testing_gene_net_derivative("Dataset1/", a,b,c,20)


# In[ ]:


NRMSE_n_file={}
    
for n in [0,10,15,20,25,50]:
    MI_by_file, X_by_file, Y_by_file, NRMSE_by_file=testing_gene_net_derivative("Dataset1/", a,b,c,n)
    
    for key in NRMSE_by_file.keys():
        NRMSE_n_file.setdefault(key,{})
        NRMSE_n_file[key][n]= NRMSE_by_file[key]
        
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    title("n="+str(n))
    plot_MI_by_file(MI_by_file)

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
title("minimum half life 15 min")
plot_MI_by_file(NRMSE_n_file)


# In[ ]:


for n in [0,10,20,50,80,100]:
    MI_by_file, X_by_file, Y_by_file, NRMSE_by_file=testing_gene_net_derivative("Dataset1/", a,b,c,n)
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    title("n="+str(n))
    plot_MI_i("DBTBS",NRMSE_by_file["DBTBS"],n)
    show()


# In[ ]:


NRMSE_15min=NRMSE_by_file


# In[ ]:


plot_MI_i("modENCODE",NRMSE_5min["modENCODE"])
plot_MI_i("modENCODE",NRMSE_15min["modENCODE"])
show()


# In[ ]:


MI_by_file, X_by_file, Y_by_file, NRMSE_by_file=testing_gene_net_derivative("Dataset1/", a,b,c,0)

