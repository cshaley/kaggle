import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing, cross_validation
from sklearn.metrics import roc_auc_score as auc
import sys
sys.path.append('/home/charlie/git/xgboost/python-package/')
import xgb_wrapper as xg
import math

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
    
    strings = [x for x in strings if x not in constants]
    
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
        X = df.drop('target',1).fillna(-1);
        y = df.target
        
	#def logplusone(p):
        #    try:
	#        p=math.log(p+1)
	#    except:
	#        pass
	#    return p

        X['zerocount'] = (X == 0).astype(int).sum(axis=1) 
	X['max'] = X.max(axis=1)
	X['median'] = X.median(axis=1)
#	p=pd.read_csv("old/xgbout-0.01-10-2000-0-45-10-0.2325-auc-0.1-1-0.4.csv").set_index("ID")
#	p['target2']=p['target']
#	p=p.drop('target',1)
#	X=X.merge(p,left_index=True,right_index=True)
	#for col in list(X.columns.values):
        #    X[col]=X[col].apply(logplusone)

        kf = cross_validation.StratifiedKFold(y, n_folds=3, shuffle=True, random_state=11)
        trscores, cvscores = [], []
        for itr, icv in kf:
            rf.fit(X.iloc[itr], y.iloc[itr])
            trscore = auc(y.iloc[itr], rf.predict_proba(X.iloc[itr])[:,1])
            cvscore = auc(y.iloc[icv], rf.predict_proba(X.iloc[icv])[:,1])
            trscores.append(trscore), cvscores.append(cvscore)
            print "TRAIN %.5f | TEST %.5f" % (np.mean(trscores), np.mean(cvscores))
        rf.fit(X,y)
    return np.mean(trscores), np.mean(cvscores)

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
        
        df=df.drop(constants,axis=1)
        
        for col in strings:
            df[col] = encoders[col].fit_transform(df[col])
	#def logplusone(p):
        #    try:
        #        p=math.log(p+1)
        #    except:
        #        pass
        #    return p

        df['zerocount'] = (df == 0).astype(int).sum(axis=1)
        df['max'] = df.max(axis=1)
        df['median'] = df.median(axis=1)

	#p=pd.read_csv("old/xgbout-0.01-10-2000-0-45-10-0.2325-auc-0.1-1-0.4.csv",index_col='ID')
        #p['target2']=p['target']
        #p=p.drop('target',1)
        #df=pd.merge(p,left_index=True,right_index=True)

        #for col in list(X.columns.values):
        #    X[col]=X[col].apply(logplusone)

        if sample_no == 0:
            submission = pd.DataFrame(rf.predict_proba(df.fillna(-1))[:,1], index=df.index, columns=['target'])
            submission.index.name = 'ID'
            submission.to_csv(output_datapath)
        else:
            submission = pd.DataFrame(rf.predict_proba(df.fillna(-1))[:,1], index=df.index, columns=['target'])
            submission.index.name = 'ID'
            with open(output_datapath,'a') as f:
                submission.to_csv(f,header=False)


train_datapath='../dl/train.csv'
test_datapath = '../dl/test.csv'
logfile = 'multicl.log'
max_cols_to_load = 3000
max_rows_to_load = 160000
constants,strings,encoders = get_strings_and_constants(train_datapath,test_datapath,max_cols_to_load)

#with open(logfile,'w') as g:
#    g.write("learning_rate	max_depth	n_boost_rounds	gamma	min_child_weight	max_delta_steps	base_score	train_score	test_score\n")

print "\nmax_delta_step = " + str(mds) + "     gamma = " + str(gm) + "    min_child_weight = " + str(mcw) + "\nlearning_rate="+str(lr)+"   max_depth="+str(md)+"   n_boost_rounds="+ str(ne)+ "\neval_metric = "+str(em)+"\nalpha = "+ str(al) + "\nlambda = "+str(la)+"\ncolsample_bytree = " + str(cs)+"\n"

    xgcl = xg.XGBoostClassifier().set_params(num_boost_round=2000,num_class=2,nthread=4,max_depth=40,learning_rate=0.01,gamma=0,min_child_weight=45,max_delta_step=10,base_score=0.2325,eval_metric='auc',colsample_bytree=0.4,alpha=0.1)#,scale_pos_weight=3.3011)
rfcl = ensemble.RandomForestClassifier(n_jobs=15, n_estimators = 200, random_state = 11)
lrcl = linear_model.LogisticRegression(max_iter=15, solver='liblinear')
etcl = ensemble.ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=11)
adacl = ensemble.AdaBoostClassifier(n_estimators=200)
gbcl = GradientBoostingRegressor(n_estimators=200, learning_rate=.03,max_depth=15, random_state=11, loss='ls')
kncl2 =  neighbors.KNeighborsRegressor(n_neighbors=2)
kncl4 =  neighbors.KNeighborsRegressor(n_neighbors=4)
kncl8 =  neighbors.KNeighborsRegressor(n_neighbors=8)
kncl16 =  neighbors.KNeighborsRegressor(n_neighbors=16)
kncl32 =  neighbors.KNeighborsRegressor(n_neighbors=32)
kncl64 =  neighbors.KNeighborsRegressor(n_neighbors=64)
kncl128 =  neighbors.KNeighborsRegressor(n_neighbors=128)
kncl256 =  neighbors.KNeighborsRegressor(n_neighbors=256)
kncl512 =  neighbors.KNeighborsRegressor(n_neighbors=512)
kncl1024 =  neighbors.KNeighborsRegressor(n_neighbors=1024)
nbcl = naive_bayes.MultinomialNB(alpha=lr)
cls=xgcl

    train_score, test_score = fit_train(train_datapath,clcl,max_rows_to_load,constants,strings,encoders)

    with open(logfile,'a') as f:
        f.write(str(lr)+"	"+str(md)+"	"+str(ne)+"	"+str(gm)+"	"+str(mcw)+"	"+str(mds)+"	"+str(bs)+"	"+str(train_score)+"	"+str(test_score) + "	"+str(em)+ "	"+str(al) + "	"+str(la)+"	"+str(cs)+"\n")

    output_datapath = 'xgboutmeta'+'-'+str(lr)+'-'+str(md)+'-'+str(ne)+'-'+str(gm)+'-'+str(mcw)+'-'+str(mds)+'-'+str(bs)+'-'+str(em)+'-'+str(al)+'-'+str(la)+'-'+str(cs)+'.csv'
    predict_test(test_datapath,output_datapath,xgcl,max_rows_to_load,constants,strings,encoders)


#https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
#https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py
