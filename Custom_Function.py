#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:16:57 2019

@author: wexu
"""

def event_to_pandas(filename):
    
    import mne
    import pandas as pd
    import numpy as np
    from config_GP_Learn import delay
    
    raw_tsss_mc=mne.io.read_raw_fif(filename,preload=True) 
    events=mne.find_events(raw_tsss_mc, stim_channel='STI101', verbose=True,min_duration=0.003)
    events[:,0]=events[:,0]+delay #fix delay
    events[:,2]=events[:,2]-events[:,1]
    events=events[(events[:,2]>100) & (events[:,2]<700)]

    #[0task][1Learnability][2V_index][3A_index][4block][5Learn_cues][6Test_correct][7Reaction_time]
    custom_events=np.empty((len(events),8))
    custom_events[:] = np.nan
    
    block=0
    for idx in range(len(events)):
        #print(events[idx])
        if events[idx,2]>100 and events[idx,2]<300:
            custom_events[idx,0]=0
            custom_events[idx,1]=1 if events[idx,2]<200 else 0
            custom_events[idx,2]=int(str(events[idx,2])[1])
            custom_events[idx,3]=int(str(events[idx,2])[2])
    
            if idx==0:            
                block=block
            elif (custom_events[idx,0]-custom_events[idx-1,0])==-1:
                block=block+1            
            custom_events[idx,4]=block
            
        elif events[idx,2]>300 and events[idx,2]<500:
            custom_events[idx,0]=1
            custom_events[idx,1]=1 if events[idx,2]<400 else 0
            custom_events[idx,2]=int(str(events[idx,2])[1])
            custom_events[idx,3]=int(str(events[idx,2])[2])
            custom_events[idx,4]=block
        elif events[idx,2]>500 and events[idx,2]<600:        
            custom_events[idx,0]=0
            custom_events[idx,4]=block
            
            if events[idx,2]==510:
                custom_events[idx,5]=1
            elif events[idx,2]==520:
                custom_events[idx,5]=2
            elif events[idx,2]==530:
                custom_events[idx,5]=3
                
        elif events[idx,2]>=600 and events[idx,2]<700:
            custom_events[idx,0]=1
            custom_events[idx,4]=block
            
            if (events[idx,2]==610 or events[idx,2]==630) & (events[idx-1,2]==600) & (events[idx-2,2]>300 and events[idx-2,2]<500):
                custom_events[idx-2,6]=1 if int(str(events[idx,2])[1])==1 else 0
                custom_events[idx-2,7]=events[idx,0]-events[idx-1,0]-100
                
    all_events=np.concatenate((events,custom_events),axis=1)
    
    df=pd.DataFrame(all_events,columns=['stim_onset','button_press','trigger_code','task','Learnability',
                                        'V_index','A_index','block','Learn_cues','Test_correct','Reaction_time'])
    
    
    Reorder_columns = ['stim_onset', 'button_press', 'trigger_code','block','task','V_index',
                       'A_index','Learnability','Learn_cues','Test_correct','Reaction_time']
    df = df.reindex(columns=Reorder_columns)
    
    #Day 1 and Day 2
    if df['block'].max()>6:    
        df =df[(df['block']>0) & (df['trigger_code']<600)] #remove practise block
    else:
        df =df[(df['trigger_code']<600)]
        df['block']=df['block']+1 #No practise block in day 2
            
    
    df['RT_Zscored'] = (df['Reaction_time'] - df['Reaction_time'].mean())/df['Reaction_time'].std(ddof=0)
    df['RT_Outlier'] = (df['RT_Zscored'] >3)
    
    df['Congruency']=np.nan
    df.loc[(df['Learnability']==1),'Congruency']=df.loc[ (df['Learnability']==1),'V_index'] == df.loc[(df['Learnability']==1),'A_index']
    
    
    
    df['V_block_learn_IDX']=np.nan
    df['A_block_learn_IDX']=np.nan
    
    df['V_block_learn_ALL']=np.nan
    df['A_block_learn_ALL']=np.nan
    
    for b in range(1,int(df['block'].max())+1):    
        
        for LB in [0,1]:
            
            AV_testing_Block=df[(df['block']==b) & (df['task']==1) & (df['Learnability']==LB)]
    
            for AV_Idx in range(len(AV_testing_Block)):
            
                V_Index=AV_testing_Block.loc[AV_testing_Block.index[AV_Idx],'V_index']
                A_Index=AV_testing_Block.loc[AV_testing_Block.index[AV_Idx],'A_index']        
                                        
                AV_testing_Block_Index =df[(df['block']==b) & (df['task']==1) & (df['Learnability']==LB) & (df['V_index']==V_Index) & (df['A_index']==A_Index)].index
                AV_learning_Block_Index=df[(df['block']==b) & (df['task']==0) & (df['Learnability']==LB) & (df['V_index']==V_Index) & (df['A_index']==A_Index)].index
    
                V_learning_Block_Index=df[(df['block']==b) & (df['task']==0) & (df['Learnability']==LB) & (df['V_index']==V_Index) & (df['A_index']==0)].index
                A_learning_Block_Index=df[(df['block']==b) & (df['task']==0) & (df['Learnability']==LB) & (df['V_index']==0) & (df['A_index']==A_Index)].index
    
                V_block_learn_IDX=df[(df['block']==b) & (df['task']==1) & (df['Learnability']==LB) & (df['V_index']==V_Index)]['Test_correct'].sum()
                A_block_learn_IDX=df[(df['block']==b) & (df['task']==1) & (df['Learnability']==LB) & (df['A_index']==A_Index)]['Test_correct'].sum()
                
                #add test_correctness filter to count for missing responses
                V_block_learn_ALL=len(df[(df['block']==b) & (df['task']==1) & (df['Learnability']==LB) & (df['V_index']==V_Index) & (df['Test_correct']>=0)]['Test_correct'])
                A_block_learn_ALL=len(df[(df['block']==b) & (df['task']==1) & (df['Learnability']==LB) & (df['A_index']==A_Index) & (df['Test_correct']>=0)]['Test_correct'])
                
                df.loc[AV_testing_Block_Index,'V_block_learn_IDX']=V_block_learn_IDX
                df.loc[AV_testing_Block_Index,'A_block_learn_IDX']=A_block_learn_IDX
    
                df.loc[AV_testing_Block_Index,'V_block_learn_ALL']=V_block_learn_ALL
                df.loc[AV_testing_Block_Index,'A_block_learn_ALL']=A_block_learn_ALL
                
                if AV_learning_Block_Index.any():                
                    df.loc[AV_learning_Block_Index,'V_block_learn_IDX']=V_block_learn_IDX
                    df.loc[AV_learning_Block_Index,'A_block_learn_IDX']=A_block_learn_IDX
    
                    df.loc[AV_learning_Block_Index,'V_block_learn_ALL']=V_block_learn_ALL
                    df.loc[AV_learning_Block_Index,'A_block_learn_ALL']=A_block_learn_ALL
                    
                    df.loc[AV_learning_Block_Index,['Test_correct','Reaction_time','RT_Zscored','RT_Outlier']]=\
                    AV_testing_Block.loc[AV_testing_Block.index[AV_Idx],['Test_correct','Reaction_time','RT_Zscored','RT_Outlier']].values
                    
    
                    if df.loc[AV_learning_Block_Index+1,'trigger_code'].isin([510,520,530]).any():
                        
                        df.loc[AV_learning_Block_Index+1,'V_block_learn_IDX']=V_block_learn_IDX
                        df.loc[AV_learning_Block_Index+1,'A_block_learn_IDX']=A_block_learn_IDX
        
                        df.loc[AV_learning_Block_Index+1,'V_block_learn_ALL']=V_block_learn_ALL
                        df.loc[AV_learning_Block_Index+1,'A_block_learn_ALL']=A_block_learn_ALL                
                        
                        df.loc[AV_learning_Block_Index+1,['Test_correct','Reaction_time','RT_Zscored','RT_Outlier']]=\
                        AV_testing_Block.loc[AV_testing_Block.index[AV_Idx],['Test_correct','Reaction_time','RT_Zscored','RT_Outlier']].values                
                
                if V_learning_Block_Index.any():                
                    df.loc[V_learning_Block_Index,'V_block_learn_IDX']=V_block_learn_IDX
                    df.loc[V_learning_Block_Index,'V_block_learn_ALL']=V_block_learn_ALL
    
                if A_learning_Block_Index.any():                
                    df.loc[A_learning_Block_Index,'A_block_learn_IDX']=A_block_learn_IDX
                    df.loc[A_learning_Block_Index,'A_block_learn_ALL']=A_block_learn_ALL
                    
    
    df['V_block_learn_index']=(df['V_block_learn_IDX']==df['V_block_learn_ALL'])
    df['A_block_learn_index']=(df['A_block_learn_IDX']==df['A_block_learn_ALL'])
    
    
    df['V_overall_learn_index']=np.nan
    df['A_overall_learn_index']=np.nan                
                    
    for LB in [0,1]:    
        for av in range(1,7):
            
            blocks=df[ (df['V_index']==av) & (df['task']==1) & (df['Learnability']==LB)]['block'].unique()                
            
            for b in blocks:            
                
                V_block_learn_index=df.loc[(df['V_index']==av) & (df['block']==b) & (df['task']==1) & (df['Learnability']==LB),'V_block_learn_index'].all()
                last_V_block_learn_index=0 if b==blocks.min() else \
                int(df.loc[(df['V_index']==av) & (df['block']==(blocks[blocks.tolist().index(b)-1])) & (df['task']==1) & (df['Learnability']==LB),'V_block_learn_index'].all())        
              
                if b==blocks.min():
                   df.loc[(df['V_index']==av) & (df['block']==b) & (df['Learnability']==LB),'V_overall_learn_index']=int(V_block_learn_index)
                   Total_V_overall_learn_index=int(V_block_learn_index)
                elif last_V_block_learn_index==0 and int(V_block_learn_index)==0:
                    V_overall_learn_index=0
                    Total_V_overall_learn_index=Total_V_overall_learn_index-1 if Total_V_overall_learn_index>1 else 0
                    df.loc[(df['V_index']==av) & (df['block']==b) & (df['Learnability']==LB),'V_overall_learn_index']=V_overall_learn_index            
                elif last_V_block_learn_index==0 and int(V_block_learn_index)==1:
                    Total_V_overall_learn_index=Total_V_overall_learn_index+1
                    V_overall_learn_index=Total_V_overall_learn_index
                    df.loc[(df['V_index']==av) & (df['block']==b) & (df['Learnability']==LB),'V_overall_learn_index']=V_overall_learn_index            
                elif last_V_block_learn_index==1 and int(V_block_learn_index)==0:
                    V_overall_learn_index=-1
                    Total_V_overall_learn_index=Total_V_overall_learn_index-1 if Total_V_overall_learn_index>1 else 0  
                    df.loc[(df['V_index']==av) & (df['block']==b) & (df['Learnability']==LB),'V_overall_learn_index']=V_overall_learn_index            
                elif last_V_block_learn_index==1 and int(V_block_learn_index)==1:
                    Total_V_overall_learn_index=Total_V_overall_learn_index+1
                    V_overall_learn_index=Total_V_overall_learn_index
                    df.loc[(df['V_index']==av) & (df['block']==b) & (df['Learnability']==LB),'V_overall_learn_index']=V_overall_learn_index
                                
                V_learning_Block_Indexs=df[(df['block']==b) & (df['task']==0) & (df['Learnability']==LB) & (df['V_index']==av) & (df['A_index']>0)].index                
                if df.loc[V_learning_Block_Indexs+1,'trigger_code'].isin([510,520,530]).all():                
                    df.loc[V_learning_Block_Indexs+1,'V_overall_learn_index']=V_overall_learn_index if b>blocks.min() else int(V_block_learn_index)            
    
    
                
            blocks=df[ (df['A_index']==av) & (df['task']==1) & (df['Learnability']==LB)]['block'].unique()                
            
            for b in blocks:            
                
                A_block_learn_index=df.loc[(df['A_index']==av) & (df['block']==b) & (df['task']==1) & (df['Learnability']==LB),'A_block_learn_index'].all()
                last_A_block_learn_index=0 if b==blocks.min() else \
                int(df.loc[(df['A_index']==av) & (df['block']==blocks[blocks.tolist().index(b)-1]) & (df['task']==1) & (df['Learnability']==LB),'A_block_learn_index'].all())        
              
                if b==blocks.min():
                   df.loc[(df['A_index']==av) & (df['block']==b) & (df['Learnability']==LB),'A_overall_learn_index']=int(A_block_learn_index)
                   Total_A_overall_learn_index=int(A_block_learn_index)
                elif last_A_block_learn_index==0 and int(A_block_learn_index)==0:
                    A_overall_learn_index=0
                    Total_A_overall_learn_index=Total_A_overall_learn_index-1 if Total_A_overall_learn_index>1 else 0
                    df.loc[(df['A_index']==av) & (df['block']==b) & (df['Learnability']==LB),'A_overall_learn_index']=A_overall_learn_index            
                elif last_A_block_learn_index==0 and int(A_block_learn_index)==1:
                    Total_A_overall_learn_index=Total_A_overall_learn_index+1
                    A_overall_learn_index=Total_A_overall_learn_index
                    df.loc[(df['A_index']==av) & (df['block']==b) & (df['Learnability']==LB),'A_overall_learn_index']=A_overall_learn_index            
                elif last_A_block_learn_index==1 and int(A_block_learn_index)==0:
                    A_overall_learn_index=-1
                    Total_A_overall_learn_index=Total_A_overall_learn_index-1 if Total_A_overall_learn_index>1 else 0  
                    df.loc[(df['A_index']==av) & (df['block']==b) & (df['Learnability']==LB),'A_overall_learn_index']=A_overall_learn_index            
                elif last_A_block_learn_index==1 and int(A_block_learn_index)==1:
                    Total_A_overall_learn_index=Total_A_overall_learn_index+1
                    A_overall_learn_index=Total_A_overall_learn_index
                    df.loc[(df['A_index']==av) & (df['block']==b) & (df['Learnability']==LB),'A_overall_learn_index']=A_overall_learn_index
                                
                A_learning_Block_Indexs=df[(df['block']==b) & (df['task']==0) & (df['Learnability']==LB) & (df['A_index']==av) & (df['V_index']>0)].index                
                if df.loc[A_learning_Block_Indexs+1,'trigger_code'].isin([510,520,530]).all():                
                    df.loc[A_learning_Block_Indexs+1,'A_overall_learn_index']=A_overall_learn_index if b>blocks.min() else int(A_block_learn_index)            
     
    
              
    df['AV_overall_learn_index']=0            
    df.loc[(df['V_overall_learn_index']>0),'AV_overall_learn_index']=df.loc[(df['V_overall_learn_index']>0),'AV_overall_learn_index']+df[(df['V_overall_learn_index']>0)]['V_overall_learn_index']
    df.loc[(df['A_overall_learn_index']>0),'AV_overall_learn_index']=df.loc[(df['A_overall_learn_index']>0),'AV_overall_learn_index']+df[(df['A_overall_learn_index']>0)]['A_overall_learn_index']
    
    return df