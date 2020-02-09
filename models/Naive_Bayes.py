import numpy as np 

class NaiveBayes:
    n_features=0
    log_priors=[]
    class_labels=[0,1]
    likelihood=[]

    #   X:features (array like), y:class labels (list), likelihood_distribution: the distribution of features, negative for continuous features,
    #   positive for categorical features, which indicates the smoothing parameter (list)
    def __init__(self, class_labels=[0,1]):
        self.class_labels=class_labels
        
    def fit(self, X, y,likelihood_distribution):
        if X.shape[0]!=len(y):
            raise ValueError("Dimensions of X and y must match.")
        else:
            _, n_features = X.shape
            priors=[0,0]
            likelihood=[]
            for j in y:
                if j==self.class_labels[0]:
                    priors[0]=priors[0]+1
                else:
                    priors[1]=priors[1]+1
            # print("priors=",priors)

            for j in range(n_features):
                if likelihood_distribution[j]>=0:
                    xd_categories=[]
                    xd_count_0=[]
                    xd_count_1=[]
                    N0=0
                    N1=0
                    for i in range(len(y)):
                        if X[i][j] not in xd_categories:
                            xd_categories.append(X[i][j])
                            if y[i]==self.class_labels[0]:
                                N0=N0+1
                                xd_count_0.append(1)
                                xd_count_1.append(0)
                            else:
                                N1=N1+1
                                xd_count_1.append(1)
                                xd_count_0.append(0)
                        else:
                            t=xd_categories.index(X[i][j])
                            if y[i]==self.class_labels[0]:
                                N0=N0+1
                                xd_count_0[t]=xd_count_0[t]+1
                            else:
                                N1=N1+1
                                xd_count_1[t]=xd_count_1[t]+1

                    alpha=likelihood_distribution[j]
                    ni=len(xd_categories)+1
                    for t in range(len(xd_count_0)):
                        # print('*',t)
                        xd_count_0[t]=np.log((xd_count_0[t]+alpha)/(N0+alpha*ni))
                        xd_count_1[t]=np.log((xd_count_1[t]+alpha)/(N1+alpha*ni))

                    #Unseen categories
                    xd_categories.append('UNK')
                    xd_count_0.append(np.log(alpha/(N0+alpha*ni)))
                    xd_count_1.append(np.log(alpha/(N1+alpha*ni)))
                    likelihood.append([xd_categories,xd_count_0,xd_count_1])

                else:    
                    xd_0=[]
                    xd_1=[]
                    for i in range(len(y)):
                        if y[i]==self.class_labels[0]:
                            xd_0.append(float(X[i][j]))
                        else:
                            xd_1.append(float(X[i][j]))
                    miu_d_0=np.mean(xd_0)
                    miu_d_1=np.mean(xd_1)
                    sigma_d_0=np.std(xd_0)
                    sigma_d_1=np.std(xd_1)

                    likelihood.append([[np.mean(xd_0),np.std(xd_0)],[np.mean(xd_1),np.std(xd_1)]])
            
            self.log_priors=np.log(priors)-np.log(len(y))
            self.n_features=n_features
            self.likelihood=likelihood

    #   X:features(array like), likelihood_distribution: the distribution of features, negative for continuous features,
    #   positive for categorical features, which indicates the smoothing parameter(list)
    def predict(self, X,likelihood_distribution):
        if X.shape[1]!=self.n_features:
            raise ValueError("Number of features should be %s."%self.n_features)
        else:
            y=[]
            for i in range(X.shape[0]): 
                y0=self.log_priors[0]
                y1=self.log_priors[1]
                for j in range(X.shape[1]): 
                    if likelihood_distribution[j]>=0:
                        if X[i][j] in self.likelihood[j][0]:
                            c=list(self.likelihood[j][0])
                            idx=c.index(X[i][j])
                            y0=y0+self.likelihood[j][1][idx]
                            y1=y1+self.likelihood[j][2][idx]
                        else:
                            y0=y0+self.likelihood[j][1][-1]
                            y1=y1+self.likelihood[j][2][-1]
                    else:
                  
                        mu_0=float(self.likelihood[j][0][0])
                        sigma_0=float(self.likelihood[j][0][1])
                        mu_1=float(self.likelihood[j][1][0])
                        sigma_1=float(self.likelihood[j][1][1])  
                        if sigma_0==0 or sigma_1==0:
                            continue
                        x=float(X[i][j])        
                        y0=y0-np.square(x-mu_0)/(2*np.square(sigma_0))-0.5*np.log(2*3.14*np.square(sigma_0))
                        y1=y1-np.square(x-mu_1)/(2*np.square(sigma_1))-0.5*np.log(2*3.14*np.square(sigma_1))

                if y0>=y1:
                    y.append(self.class_labels[0])
                else:
                    y.append(self.class_labels[1])
                
        return y

    #y:true labels, y1:predicted labels
    def evaluate_acc(self, y,y1):
        acc=0
        if len(y)!=len(y1):
            raise ValueError('Lengths of two input arrays must match.')
        else:
            s=len(y)
            T=0;
            for i in range(s):
                if y[i]==y1[i]:
                    T=T+1
                    acc=T/s
            return acc
