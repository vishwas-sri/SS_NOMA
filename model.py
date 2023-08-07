import numpy as np
from sklearn import metrics as mt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from scipy import special
import tensorflow as tf
tf.autograph.set_verbosity(1)

Pfa_target=[x/10000.0 for x in range(25,10000,25)]
# Pfa_target=[0.4,0.3,0.2,0.1]
class Classification:
    #print(np.concatenate((y_pred.reshpipape(len(y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))
    def __init__(self, X_train=None,y_train=None,X_test=None,y_test=None,samples=None,SU=None,X_test_2=None):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.samples=samples
        self.SU=SU
        self.X_test_2=X_test_2
        # self.SNR=SNR
        # sc= StandardScaler()
        # self.X_train = sc.fit_transform(self.X_train) 
        # self.X_test = sc.transform(self.X_test)
        # self.X_combined = np.r_[self.X_train, self.X_test]
        # self.y_combined = np.r_[self.y_train, self.y_test] 
        # self.y_train=self.y_train.reshape(-1)
        # df_train.info()
    

    # def main(self):
    #     val=[]

    def Linear_SVM(self):
        classifier = SVC()
        types = 'LinearSVM'
        # marker = "X"
         
        # parameters = [{'C': [0.2, 0.4, 0.6, 0.8, 1],#, 1.2, 1.4,10, 100, 1000],
        #                 'gamma': [1, 0.1, 0.01],#, 0.001, 0.0001],
        #                 'kernel': ['linear'], 'probability':[True]}]
        parameters = [{'C': [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4,10, 100, 1000],
                        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                    'kernel': ['linear'], 'probability':[True]}]
        grid_search = GridSearchCV(
            estimator=classifier, param_grid=parameters, scoring='accuracy', n_jobs=-1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        # best_accuracy = grid_search.best_score_
        # best_parameters = grid_search.best_params_
        
        # for actual prediction
        y_pred = grid_search.predict(self.X_test)
        
        # # for ROC, probability prediction
        # y_pred2=grid_search.predict_proba(self.X_test)
        # y_pred2=y_pred2[:,1]
        
        # # print(1-np.sum(y_pred)/len(y_pred))
        # # print(self.y_test[0:10])
        # # cm = mt.confusion_matrix(y_test, y_pred)
        # # accuracy = mt.accuracy_score(y_test, y_pred)
        # fpr, tpr, _ = mt.roc_curve(self.y_test,  y_pred2)
        # # Create SVM classifier with linear kernel
        # # clf = svm.SVC(kernel='linear')
        # # # clf = svm.SVC(kernel='rbf') # gaussian

        # # # Train the model using the training data
        # # clf.fit(X_train, y_train)

        # # # Make predictions on the test data
        # # y_pred = clf.predict(X_test)

        # # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

        # auc = mt.auc(fpr, tpr)
        # mark = int((len(fpr))*0.037)
        
        # return fpr, tpr, auc, types, y_pred#mark
        return y_pred
    
    

    def Gaussian_SVM(self):
        classifier = SVC()
        types = 'GaussianSVM'
        # marker = "X"
        # parameters = [{'C': [0.1, 1, 10, 100, 1000],  
        #                 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        #             'kernel': ['rbf'], 'probability':[True]}]
        parameters = [{'C': [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4,10, 100, 1000],
                        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                        'kernel': ['rbf'], 'probability':[True]}]
        grid_search = GridSearchCV(
            estimator=classifier, param_grid=parameters, scoring='accuracy', n_jobs=-1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        # best_accuracy = grid_search.best_score_
        # best_parameters = grid_search.best_params_
        # y_pred = grid_search.predict(self.X_test)
        y_pred2=grid_search.predict_proba(self.X_test)
        y_pred2=y_pred2[:,1]
        
        fpr, tpr, _ = mt.roc_curve(self.y_test,  y_pred2)
        
        auc = mt.auc(fpr, tpr)
        # mark = int((len(fpr))*0.037)
        return fpr, tpr, auc, types #mark
    
    def Logistic(self):
        classifier=LogisticRegression()
        types='Logistic'
        # marker="o"
        parameters =[{'C': [0.2, 0.4, 0.6, 0.8, 1],# 1.2, 1.4,10, 100, 1000], 
                      'max_iter':[1000],'solver': ['newton-cg','lbfgs','sag'], 'penalty': ['l2']},
                     {'C': [0.2, 0.4, 0.6, 0.8, 1],# 1.2, 1.4,10, 100, 1000],
                      'max_iter':[1000], 'solver': ['saga','liblinear'], 'penalty': ['l1','l2']}]
        grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = 'accuracy',n_jobs = -1, cv=10,verbose=0)
        grid_search.fit(self.X_train, self.y_train)
        # best_accuracy = grid_search.best_score_
        # best_parameters=grid_search.best_params_
        # y_pred=grid_search.predict(self.X_test)
        # cm = confusion_matrix(self.y_test, y_pred)
        # accuracy=accuracy_score(self.y_test, y_pred)
        # cm2=confusion_matrix(self.y_train, grid_search.predict(self.X_train))
        # output=[type,accuracy*100,cm,cm2,best_accuracy*100,best_parameters]
        y_pred2=grid_search.predict_proba(self.X_test)
        y_pred2=y_pred2[:,1]
        fpr, tpr, _ = mt.roc_curve(self.y_test,  y_pred2)
        auc = mt.auc(fpr,tpr)
        return[fpr,tpr,auc,types]
    
    def NaiveBayes(self):
        classifier = GaussianNB()
        types='NaiveBayes'
        # marker="*"
        parameters =[{'var_smoothing':[1e-9]}]
        grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = 'accuracy',n_jobs = -1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        # best_accuracy = grid_search.best_score_
        # best_parameters=grid_search.best_params_
        # y_pred=grid_search.predict(self.X_test)
        # cm = confusion_matrix(self.y_test, y_pred)
        # accuracy=accuracy_score(self.y_test, y_pred)
        # cm2=confusion_matrix(self.y_train, grid_search.predict(self.X_train))
        # plt=1
        # # plt=plot_learning_curve(estimator=classifier,title=type,X=self.X_combined, y=self.y_combined)
        # output=[type,accuracy*100,cm,cm2,best_accuracy*100,best_parameters,plt]
        
        y_pred2=grid_search.predict_proba(self.X_test)
        y_pred2=y_pred2[:,1]
        fpr, tpr, _ = mt.roc_curve(self.y_test,  y_pred2)
        auc = mt.auc(fpr,tpr)
        return[fpr,tpr,auc,types]
    
    def MLP(self):
        types='MLP'
        # marker="s"
        ann = tf.keras.models.Sequential()

        ann.add(tf.keras.layers.Dense(units=self.SU, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=len(self.y_train), activation='relu'))
        ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        ann.fit(self.X_train, self.y_train, epochs =500,verbose=0)
        y_pred2=ann.predict(self.X_test)
        y_pred2=y_pred2.flatten()
        fpr, tpr, _ = mt.roc_curve(self.y_test,  y_pred2)
        auc = mt.auc(fpr,tpr)
        return[fpr,tpr,auc,types]
    
        

    def S1(self):
        
        # Pfa_target=[x/10000.0 for x in range(25,10000,25)]
        tpr=[]
        fpr=[]
        types="S1"
        # marker="1"
        for i in range(len(Pfa_target)):
            # alpha=1-Pfa_target[i]
            lambd = 2*special.gammainccinv(self.samples/2,Pfa_target[i])/self.samples
            # lambd = 2*special.gammainccinv(self.SU/2,Pfa_target[i])/self.SU
            y_pred=np.array(self.X_test[:,0]>=lambd, dtype=int)
            
            tn=np.sum(np.logical_not(self.y_test)&np.logical_not(y_pred))
            tp=np.sum(np.logical_and(self.y_test, y_pred))
            fn=np.sum(np.logical_and(self.y_test,np.logical_not(y_pred)))
            fp=np.sum(np.logical_and(np.logical_not(self.y_test),y_pred))
            tpr.append(tp/(tp+fn))
            fpr.append(fp/(fp+tn))
        # print(fpr)
        # print(tpr)
        auc = mt.auc(fpr,tpr)
        tpr = np.array(tpr)
        fpr = np.array(fpr)
        return[fpr,tpr,auc,types]
    
    def S2(self):
        
        # Pfa_target=[x/10000.0 for x in range(25,10000,25)]
        tpr=[]
        fpr=[]
        types="S2"
        # marker="1"
        for i in range(len(Pfa_target)):
            # alpha=1-Pfa_target[i]
            lambd = 2*special.gammainccinv(self.samples/2,Pfa_target[i])/self.samples
            # lambd = 2*special.gammainccinv(self.SU/2,Pfa_target[i])/self.SU
            y_pred=np.array(self.X_test[:,1]>=lambd, dtype=int)
            
            tn=np.sum(np.logical_not(self.y_test)&np.logical_not(y_pred))
            tp=np.sum(np.logical_and(self.y_test, y_pred))
            fn=np.sum(np.logical_and(self.y_test,np.logical_not(y_pred)))
            fp=np.sum(np.logical_and(np.logical_not(self.y_test),y_pred))
            tpr.append(tp/(tp+fn))
            fpr.append(fp/(fp+tn))
        # print(fpr)
        # print(tpr)
        auc = mt.auc(fpr,tpr)
        tpr = np.array(tpr)
        fpr = np.array(fpr)
        return[fpr,tpr,auc,types]
    
    def S3(self):
        
        # Pfa_target=[x/10000.0 for x in range(25,10000,25)]
        tpr=[]
        fpr=[]
        types="S3"
        # marker="1"
        for i in range(len(Pfa_target)):
            # alpha=1-Pfa_target[i]
            lambd = 2*special.gammainccinv(self.samples/2,Pfa_target[i])/self.samples
            # lambd = 2*special.gammainccinv(self.SU/2,Pfa_target[i])/self.SU
            y_pred=np.array(self.X_test[:,2]>=lambd, dtype=int)
            
            tn=np.sum(np.logical_not(self.y_test)&np.logical_not(y_pred))
            tp=np.sum(np.logical_and(self.y_test, y_pred))
            fn=np.sum(np.logical_and(self.y_test,np.logical_not(y_pred)))
            fp=np.sum(np.logical_and(np.logical_not(self.y_test),y_pred))
            tpr.append(tp/(tp+fn))
            fpr.append(fp/(fp+tn))
        # print(fpr)
        # print(tpr)
        auc = mt.auc(fpr,tpr)
        tpr = np.array(tpr)
        fpr = np.array(fpr)
        return[fpr,tpr,auc,types]
    
    def OR(self):
        # Pfa_target=[x/10000.0 for x in range(25,10000,25)]
        tpr=[]
        fpr=[]
        types="OR"
        # marker="v"
        for i in range(len(Pfa_target)):
            # alpha=1-Pfa_target[i]
            lambd = 2*special.gammainccinv(self.samples/2,Pfa_target[i])/self.samples
            # lambd = 2*special.gammainccinv(self.SU/2,Pfa_target[i])/self.SU
            y_pred=np.array(np.sum(self.X_test>=lambd,1)>0, dtype=int)    

            tn=np.sum(np.logical_not(self.y_test)&np.logical_not(y_pred))
            tp=np.sum(np.logical_and(self.y_test, y_pred))
            fn=np.sum(np.logical_and(self.y_test,np.logical_not(y_pred)))
            fp=np.sum(np.logical_and(np.logical_not(self.y_test),y_pred))
            tpr.append(tp/(tp+fn))
            fpr.append(fp/(fp+tn))
                      
        
        # print(fpr)
        # print(tpr)
        auc = mt.auc(fpr,tpr)
        tpr = np.array(tpr)
        fpr = np.array(fpr) 
        return[fpr,tpr,auc,types]
    
    def AND(self):
        # Pfa_target=[x/10000.0 for x in range(25,10000,25)]
        tpr=[]
        fpr=[]
        types="AND"
        # marker=">"
        for i in range(len(Pfa_target)):
            # alpha=1-Pfa_target[i]
            lambd = 2*special.gammainccinv(self.samples/2,Pfa_target[i])/self.samples
            # lambd = 2*special.gammainccinv(self.SU/2,Pfa_target[i])/self.SU
            y_pred=np.array(np.sum(self.X_test>=lambd,1)==self.SU, dtype=int)
            
            tn=np.sum(np.logical_not(self.y_test)&np.logical_not(y_pred))
            tp=np.sum(np.logical_and(self.y_test, y_pred))
            fn=np.sum(np.logical_and(self.y_test,np.logical_not(y_pred)))
            fp=np.sum(np.logical_and(np.logical_not(self.y_test),y_pred))
            tpr.append(tp/(tp+fn))
            fpr.append(fp/(fp+tn))
        
        auc = mt.auc(fpr,tpr)
        tpr = np.array(tpr)
        fpr = np.array(fpr)
        return[fpr,tpr,auc,types]
    
    def MRC(self):
        # Pfa_target=[x/10000.0 for x in range(25,10000,25)]
        tpr=[]
        fpr=[]
        types="MRC"
        # marker="<"
        for i in range(len(Pfa_target)):
            # alpha=1-Pfa_target[i]
            lambd = 2*special.gammainccinv(self.samples/2,Pfa_target[i])/self.samples
            # lambd = 2*special.gammainccinv(self.SU/2,Pfa_target[i])/self.SU
            y_pred=np.array(np.sum(self.X_test_2,1)>lambd, dtype=int)
            
            tn=np.sum(np.logical_not(self.y_test)&np.logical_not(y_pred))
            tp=np.sum(np.logical_and(self.y_test, y_pred))
            fn=np.sum(np.logical_and(self.y_test,np.logical_not(y_pred)))
            fp=np.sum(np.logical_and(np.logical_not(self.y_test),y_pred))
            tpr.append(tp/(tp+fn))
            fpr.append(fp/(fp+tn))
        # print(fpr)
        # print(tpr)
        auc = mt.auc(fpr,tpr)
        tpr = np.array(tpr)
        fpr = np.array(fpr)
        return[fpr,tpr,auc,types]
