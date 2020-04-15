#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
from sklearn.metrics import mean_squared_error


# In[ ]:


class MyLinearRegression:
    """
        Arguments:
            X -- 
            y --
        Calculates slope&intercept using OLS
    """
    def __init__(self,X,y):
        self.fit(X,y)#awful,I know
    
    def fit(self,X,y):

        self.X=X
        self.y=y
        
        self.n = self.X.shape[0]
        self.slope = self.calc_b0()
        self.intercept = self.calc_b1()
        self.predict()    
    
    def calc_b0(self):#slope
        return ((self.X-self.X.mean())*(self.y-self.y.mean())).sum()/((self.X-self.X.mean())**2).sum()

    def calc_b1(self):#intercept
        b = self.y.sum()-self.slope*self.X.sum()
        return b/self.X.shape[0]
    
    def predict(self,X_test=None):
        """
            X_test -- used to make predictions for custom data
        """
        if X_test is not None:
            return X_test * self.slope + self.intercept 
        self.pred = self.X * self.slope + self.intercept
        return self.pred
    
    def cov(self):
        if not hasattr(self,'pred'):
            self.predict()    
        res = (self.y - self.y.mean())*(self.pred - self.pred.mean())
        return res.sum()/(self.y.shape[0]-1)
    
    def corr(self):
        if not hasattr(self,'pred'):
            self.predict()
        k = 1 if self.slope>0 else -1
            
        return k * self.cov()/(self.y.std()*self.pred.std())
    
    def r_squared(self):
        if not hasattr(self,'pred'):
            self.predict()
        return np.var(self.pred)/np.var(self.y)
    
    def su_squared(self):
        e = self.y-self.pred
        self.su_2 = self.n*np.var(e)/(self.n-2)
    
    def calculate_sp(self):#standard error
        if not hasattr(self,'su_2'):
            self.su_squared()
            
        sp_dict = {}
        
        n,mean,var = self.n,self.X.mean(),np.var(self.X)
        #slope
        sp_dict['slope'] = np.sqrt(self.su_2/(n*var))
        #intercept
        sp_dict['intercept'] = np.sqrt(self.su_2*(1+mean**2/var)/n)
        #y_val        
        #sp_dict['y_pred'] = np.sqrt(self.su_2/n*((n+1+(self.X-mean)**2)/var))
        sp_dict['y_pred'] = self.sp_y_pred(self.X,self.pred)
        
        self.sp_dict = sp_dict
        return sp_dict
    
    def sp_y_pred(self,x,y_pred):
        mean, var = self.X.mean(), np.var(self.X)
        #n = y_pred.shape[0]
        n = self.n
        
        a = self.su_2/n
        b = (1+(x-mean)**2)/var
        return np.sqrt(a*b)
    
    def confidence_interval(self,t_crit):
        if not hasattr(self,'sp_dict'):
            self.calculate_sp()
        self.ci_values = {
            'slope': self.sp_dict['slope'] * t_crit,
            'intercept': self.sp_dict['intercept'] * t_crit,
            'y_pred': self.sp_dict['y_pred'].apply(lambda x: x*t_crit)
        }
#        self.ci_values = {k:self.sp_dict[k]*t_crit for k in self.sp_dict.keys()}
        #self.confidence_intervals = {(self.)}
        
        return self.ci_values
        
    def __str__(self):
        return f'b0 = {self.slope:.4f}, b1 = {self.intercept:.4f}, r = {self.corr():.4f}, r_squared={self.r_squared():.4f}'
    
    def calcualte_interval(self, val,sp_val,t_crit):
        if not hasattr(self,'sp_dict'):
            self.calculate_sp()
            
        return (val-sp_val*t_crit,val+sp_val*t_crit)
    
    def get_X(self):
        return self.X
    
    def get_Y(self):
        return self.Y
    
    def get_slope(self):
        return self.slope
    
    def get_intercept(self):
        return self.intercept   
    
    def get_pred(self):
        return self.pred
    
    def get_conf_int_values(self):
        return self.ci_values
    
    def get_sp_dict(self):
        return self.sp_dict


# In[ ]:


def cov(x,y):
    res = (x -x.mean())*(y-y.mean())
    return res.sum()/(x.shape[0]-1)

def corr(x,y):
    return cov(x,y)/(x.std()*y.std())

def alpha(x,y,beta):
    return y-x*beta

def beta(x,y):
    return cov(x,y)/np.var(x)

def r_squared(y,y_pred):
    return np.var(y_pred)/np.var(y)


# In[76]:


def t_val(val,h,se):
    return (val-h)/se


# In[77]:


def MSR(y,y_pred):
    return ((y_pred-y.mean())**2).sum()

def F_val_alt(y,y_pred):
    return MSR(y,y_pred)/mean_squared_error(y,y_pred)

def F_val(y,y_pred):
    n=y.shape[0]
    R_2 = r_squared(y,y_pred)
    return (R_2*(n-2))/((1-R_2))

def t_val_pred(y,y_pred):
    r = corr(y,y_pred)
    r_s = r_squared(y,y_pred)
    k = y.shape[0]-2
    return  r * np.sqrt(k/(1-r_s))

def evaluate_hypothesis(val,crit):
    sign = '>' if val>crit else '<'
    msg = 'hypothesis rejected' if val>crit else 'hypothesis must be examined further'

    return sign,msg


# In[73]:


def s_u_squared(y,y_pred):
    e = y - y_pred
    return y.shape[0]*e.var()/(y.shape[0]-2)

def s_p(x,y,y_pred):
    s_2 = s_u_squared(y,y_pred)
    k = 1/x.var()
    n = x.shape[0]
    
    delta = x-x.mean()
    
    res = s_2*(n+1+k*(delta**2))/n
    return np.sqrt(res)

def calcualte_confidence_bound(x,y,y_pred,t_value):
    sp = s_p(x,y,y_pred)
    return sp*t_value


def confidence_interval_final(X,y,y_pred,t_crit = 1.96):
    n,e = X.shape[0],y - y_pred
    su_2 = n * np.var(e) / (n-2)

    se = np.sqrt(su_2 * (n + 1 + (X-X.mean())**2 / X.var())/ n)
    delta = se * t_crit
    return y_pred - delta, y_pred + delta

# In[ ]:




