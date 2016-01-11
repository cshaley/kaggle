import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing, cross_validation, neighbors
from sklearn.metrics import roc_auc_score as auc
import sys
sys.path.append('/home/charlie/git/xgboost/python-package/')
import xgb_wrapper as xg
import math
import time

basepath="/home/charlie/kaggle/springleaf/"

def get_strings_and_constants(train_datapath,test_datapath,max_cols_to_load):
#Load in one (or more) columns at a time using a for loop that skips when pandas does read_csv
#While loading in one at a time, keep a list
#list of cols to remove (constant)
#list of columns to encode as strings
    x=''
    with open (train_datapath,'r') as f:
        x = f.readline().strip()
    cols=x.split(',')

    constants=[]
    strings=[]
    encoders = {}

    print "Num iterations: " + str(int(np.ceil((len(cols))/max_cols_to_load))+1)
    for col in range(int(np.ceil((len(cols))/max_cols_to_load))+1):
        print "iteration number " + str(col)
        strings_bool=[]
        mask = np.zeros(len(cols),dtype=bool)
        mask[col*max_cols_to_load:min(col*max_cols_to_load+max_cols_to_load,len(cols))]=1
        use_cols = [np.array(cols)[mask][i].replace("\'","").replace('\"','') for i in range(len(np.array(cols)[mask]))]
        df=pd.read_csv(train_datapath,sep=',',usecols=use_cols)
        nunique = pd.Series([df[colu].nunique() for colu in df.columns], index = df.columns)
        constants += nunique[nunique<2].index.tolist()
        strings_bool = df.dtypes == 'object';
        strings_bool = strings_bool[strings_bool].index.tolist();
        strings += strings_bool
#    datecollist=['VAR_0075','VAR_0166','VAR_0167','VAR_0168','VAR_0169','VAR_0156','VAR_0157','VAR_0158','VAR_0159','VAR_0176','VAR_0177','VAR_0178','VAR_0179','VAR_0217']
    strings = [x for x in strings if x not in constants]# and x not in datecollist]
    
    for entry in strings:
        encoders[entry] = preprocessing.LabelEncoder()

    print "Constant columns: " + ",".join(constants) + "\n"
    print "String columns: " + ",".join(strings)
    return constants,strings,encoders

#Call a function to load max_no_rows into a df and use the lists to parse it.
def fit_train(datapath,rf,max_no_rows,constants,strings,encoders):
    print("Loading Train Data")
    line_count=0
    with open(datapath,'r') as f:
        for line in f:
            line_count+=1
    
    num_iter = max(int(np.ceil(line_count/max_no_rows)),1)
    print "Number of Iterations: " + str(num_iter)
    #need to switch order of first two ifs
    for sample_no in range(num_iter):
        print "Load & Train iteration " + str(sample_no)
        if sample_no == num_iter-1:
            df = pd.read_csv(datapath,skiprows=range(1,sample_no*max_no_rows)).set_index("ID")
        elif sample_no==0:
            df = pd.read_csv(datapath,skiprows=range((sample_no+1)*max_no_rows,line_count)).set_index("ID")
        elif sample_no>0 and sample_no<num_iter-1:
            df = pd.read_csv(datapath,skiprows=range(1,sample_no*max_no_rows)+range((sample_no+1)*max_no_rows,line_count)).set_index("ID")
        else:
            print "something is wrong"
            system.exit(1)
            #df = pd.read_csv(datapath).set_index("ID")
        #df=df.reindex(np.random.permutation(df.index))
        
        df=df.drop(constants,axis=1)
        for col in strings:
            try:
                df[col] = encoders[col].fit_transform(df[col])
            except:
                del df[col]
        X = df.drop('target',1)
        #datecollist=['VAR_0075','VAR_0166','VAR_0167','VAR_0168','VAR_0169','VAR_0156','VAR_0157','VAR_0158','VAR_0159','VAR_0176','VAR_0177','VAR_0178','VAR_0179','VAR_0217']
        #for col in datecollist:
        #    try:
        #        X[col+'weekday']=pd.to_datetime(X[col],format='%d%b%y:%H:%M:%S').dt.weekday
        #        X[col+'weekofyear']=pd.to_datetime(X[col],format='%d%b%y:%H:%M:%S').dt.weekofyear
        #        X[col+'dayofyear']=pd.to_datetime(X[col],format='%d%b%y:%H:%M:%S').dt.dayofyear
        #        X[col+'monthofyear']=pd.to_datetime(X[col],format='%d%b%y:%H:%M:%S').dt.month
        #        X[col+'quarter']=pd.to_datetime(X[col],format='%d%b%y:%H:%M:%S').dt.quarter
        #    except:
        #        pass
        X=X.fillna(-1)

        y = df.target
       
        ndf = pd.DataFrame(index=X.index) 
        def logplusmorethanone(p):
            try:
                p=math.log(p+10.1)
            except:
                pass
            return p
        
        ndf['zerocount'] = (X == 0).astype(int).sum(axis=1) 
        ndf['max'] = X.max(axis=1)
        ndf['median'] = X.median(axis=1)
        ndf['stdev']=X.std(axis=1)
        ndf['mad']=X.mad(axis=1)
        ndf['kurtosis']=X.kurt(axis=1)
        ndf['skew']=X.skew(axis=1)
        ndf['sem']=X.sem(axis=1)
        ndf['nacount'] = (X == -1).astype(int).sum(axis=1) 
        ndf['xtremelowcount'] = (X <= -1000).astype(int).sum(axis=1)
        ndf['xtremehighcount'] = (X >= 1000).astype(int).sum(axis=1)
        ndf['onecount'] = (X == 1).astype(int).sum(axis=1)

        p=pd.read_csv("old/xgbout-0.01-10-2000-0-45-10-0.2325-auc-0.1-1-0.4.csv").set_index("ID")
        p['target2']=p['target']
        p=p.drop('target',1)
        X1=pd.DataFrame(index=X.index)
        for col in list(X.columns.values):
            X1[col]=X[col].apply(logplusmorethanone)

      #  ndf['prod']=X1.prod(axis=1)
        X=X.join(ndf).join(p).fillna(-1).replace([np.inf, -np.inf], -1)
      
        #X=X.merge(p,left_index=True,right_index=True)
        #kf = cross_validation.StratifiedKFold(y, n_folds=3, shuffle=True, random_state=11)
        #trscores, cvscores = [], []
        #for itr, icv in kf:
        #    rf.fit(X.iloc[itr], y.iloc[itr])
        #    trscore = auc(y.iloc[itr], rf.predict(X.iloc[itr])[:,1])
        #    cvscore = auc(y.iloc[icv], rf.predict(X.iloc[icv])[:,1])
        #    trscores.append(trscore), cvscores.append(cvscore)
        #    print "TRAIN %.5f | TEST %.5f" % (np.mean(trscores), np.mean(cvscores))
        rf.fit(X,y)
    return None,None#np.mean(trscores), np.mean(cvscores)

def predict_test(datapath,output_datapath,rf,max_no_rows,constants,strings,encoders):
    print("Loading Test Data")
    line_count=0
    with open(datapath,'r') as f:
        for line in f:
            line_count+=1
    
    num_iter = int(np.ceil(line_count/max_no_rows))
    print "Number of Iterations: " + str(num_iter)
    #need to switch order of first two ifs
    for sample_no in range(max(num_iter,1)):
        print "Load & Predict iteration " + str(sample_no)
        if sample_no == num_iter-1:
            df = pd.read_csv(datapath,skiprows=range(1,sample_no*max_no_rows)).set_index("ID")
        elif sample_no==0:
            df = pd.read_csv(datapath,skiprows=range((sample_no+1)*max_no_rows,line_count)).set_index("ID")
        elif sample_no>0 and sample_no<num_iter-1:
            df = pd.read_csv(datapath,skiprows=range(1,sample_no*max_no_rows)+range((sample_no+1)*max_no_rows,line_count)).set_index("ID")
        else:
            print "something is wrong"
            system.exit(1)
            #df = pd.read_csv(datapath).set_index("ID")
        #df=df.set_index("ID")
        
        df=df.drop(constants,axis=1).fillna(-1)
       
        for col in strings:
            df[col] = encoders[col].fit_transform(df[col])

        def logplusmorethanone(p):
            try:
                p=math.log(p+10)
            except:
                pass
            return p
        
        #datecollist=['VAR_0075','VAR_0166','VAR_0167','VAR_0168','VAR_0169','VAR_0156','VAR_0157','VAR_0158','VAR_0159','VAR_0176','VAR_0177','VAR_0178','VAR_0179','VAR_0217']
        #for col in datecollist:
        #    try:
        #        df[col+'weekday']=pd.to_datetime(df[col],format='%d%b%y:%H:%M:%S').dt.weekday
        #        df[col+'weekofyear']=pd.to_datetime(df[col],format='%d%b%y:%H:%M:%S').dt.weekofyear
        #        df[col+'dayofyear']=pd.to_datetime(df[col],format='%d%b%y:%H:%M:%S').dt.dayofyear
        #        df[col+'monthofyear']=pd.to_datetime(df[col],format='%d%b%y:%H:%M:%S').dt.month
        #        df[col+'quarter']=pd.to_datetime(df[col],format='%d%b%y:%H:%M:%S').dt.quarter
        #    except:
        #        pass


        ndf=pd.DataFrame(index=df.index)
        ndf['zerocount'] = (df == 0).astype(int).sum(axis=1)
        ndf['max'] = df.max(axis=1)
        ndf['median'] = df.median(axis=1)
        ndf['stdev']=df.std(axis=1)
        ndf['mad']=df.mad(axis=1)
        ndf['kurtosis']=df.kurt(axis=1)
        ndf['skew']=df.skew(axis=1)
        ndf['sem']=df.sem(axis=1)
        ndf['nacount'] = (df == -1).astype(int).sum(axis=1) 
        ndf['xtremelowcount'] = (df <= -1000).astype(int).sum(axis=1) 
        ndf['xtremehighcount'] = (df >= 1000).astype(int).sum(axis=1) 
        ndf['onecount'] = (df == 1).astype(int).sum(axis=1) 
        

#        datecollist=['VAR_0075','VAR_0166','VAR_0167','VAR_0168','VAR_0169','VAR_0156','VAR_0157','VAR_0158','VAR_0159','VAR_0176','VAR_0177','VAR_0178','VAR_0179','VAR_0217']
#        for col in datecollist:
#            df[col+'weekday']=pd.to_datetime(df[col],format='%d%b%y:%H:%M:%S').dt.weekday
#            df[col+'weekofyear']=pd.to_datetime(df[col],format='%d%b%y:%H:%M:%S').dt.weekofyear
#            df[col+'dayofyear']=pd.to_datetime(df[col],format='%d%b%y:%H:%M:%S').dt.dayofyear
#            df[col+'monthofyear']=pd.to_datetime(df[col],format='%d%b%y:%H:%M:%S').dt.month
#            df[col+'quarter']=pd.to_datetime(df[col],format='%d%b%y:%H:%M:%S').dt.quarter
        
        p=pd.read_csv("old/xgbout-0.01-10-2000-0-45-10-0.2325-auc-0.1-1-0.4.csv").set_index("ID")
        p['target2']=p['target']
        p=p.drop('target',1)
        X1=pd.DataFrame(index=df.index)
        for col in list(df.columns.values):
            X1[col]=df[col].apply(logplusmorethanone)
        #ndf['prod']=X1.prod(axis=1)
        df=df.join(ndf).join(p).fillna(-1).replace([np.inf, -np.inf], -1)
        #X=X.merge(p,left_index=True,right_index=True)

        if sample_no == 0:
            submission = pd.DataFrame(rf.predict(df), index=df.index)
            submission.index.name = 'ID'
            submission.to_csv(output_datapath)
        else:
            submission = pd.DataFrame(rf.predict(df), index=df.index)
            submission.index.name = 'ID'
            with open(output_datapath,'a') as f:
                submission.to_csv(f,header=False)



train_datapath=basepath+'dl/train.csv'
test_datapath = basepath+'dl/test.csv'
logfile = basepath+'runlog.log'
max_cols_to_load = 3000
max_rows_to_load = 160000
constants,strings,encoders = get_strings_and_constants(train_datapath,test_datapath,max_cols_to_load)

learning_rates=[.01]
max_depth = [40]
n_estimators= [2000]
gammas=[0]
min_child_weights = [45]
max_delta_steps = [10]
base_scores=[0.2325]
eval_metrics=['rmse']
alphas = [0.1]
lambdas = [1]
col_subsamples=[0.4]
#with open(logfile,'w') as g:
#    g.write("thingy	other_thingy	rain_score	test_score	time taken in minutes\n")

for gm in gammas:
    for mcw in min_child_weights:
        for mds in max_delta_steps:
            for lr in learning_rates:
                for md in max_depth:
                    for ne in n_estimators:
                        for bs in base_scores:
                            for em in eval_metrics:
                                for al in alphas:
                                    for la in lambdas:
                                        for cs in col_subsamples:
                                            print "\nmax_delta_step = " + str(mds) + "     gamma = " + str(gm) + "    min_child_weight = " + str(mcw) + "\nlearning_rate="+str(lr)+"   max_depth="+str(md)+"   n_boost_rounds="+ str(ne)+ "\neval_metric = "+str(em)+"\nalpha = "+ str(al) + "\nlambda = "+str(la)+"\ncolsample_bytree = " + str(cs)+"\n"
                                            start_time1=time.time()
                                            start_time=time.time()
                                            xgcl = xg.XGBoostClassifier().set_params(num_boost_round=ne,num_class=2,nthread=4,max_depth=md,learning_rate=lr,gamma=gm,min_child_weight=mcw,max_delta_step=mds,base_score=bs,eval_metric=em,colsample_bytree=cs,alpha=al)


                                            train_score, test_score = fit_train(train_datapath,xgcl,max_rows_to_load,constants,strings,encoders)
                                            print "It took " +str((time.time()-start_time1)/60.0)+" minutes to execute."
                                            with open(logfile,'a') as f:
                                                f.write(str(lr)+"	"+str(md)+"	"+str(ne)+"	"+str(gm)+"	"+str(mcw)+"	"+str(mds)+"	"+str(bs)+"	"+str(train_score)+"	"+str(test_score) + "	"+str(em)+ "	"+str(al) + "	"+str(la)+"	"+str(cs)+"	"+str((time.time()-start_time)/60.0)+"\n")

                                            output_datapath = 'xgboutmeta'+'-'+str(lr)+'-'+str(md)+'-'+str(ne)+'-'+str(gm)+'-'+str(mcw)+'-'+str(mds)+'-'+str(bs)+'-'+str(em)+'-'+str(al)+'-'+str(la)+'-'+str(cs)+'.csv'
                                            predict_test(test_datapath,output_datapath,xgcl,max_rows_to_load,constants,strings,encoders)


#https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
#https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py
