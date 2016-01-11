import pandas as pd
import numpy as np

datalist=['old/ada_out.csv','xgb_79319.csv']
weightlist=[0.35,0.65]
outfile = 'wtavg.csv'
def weighted_average(datalist,weightlist):
    outdf=pd.read_csv(datalist[0])
    outdf['target']=outdf['target']*weightlist[0]
    for path in range(1,len(datalist)):
        x=pd.read_csv(datalist[path])
	x['target']=np.round(x['target'])
	outdf['target']=outdf['target']+x['target']*weightlist[path]
    outdf.to_csv(outfile,index=False)

weighted_average(datalist,weightlist)


