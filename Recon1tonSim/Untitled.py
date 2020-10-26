#!/usr/bin/env python
# coding: utf-8

# In[12]:


import ROOT as root
import numpy as np


# In[20]:


tTruth = root.TChain("SimTriggerInfo")
tTruth.Add('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.600_z.root')
# Loop for event
cnt = 0
for EID, event in enumerate(tTruth):
    if(len(event.PEList)==0):
        pass
    else:
        for truthinfo in event.truthList:
            E =  truthinfo.EkMerged
            x =  truthinfo.x
            y =  truthinfo.y
            z =  truthinfo.z
            for px in truthinfo.PrimaryParticleList:
                pxx = px.px
                pyy = px.py
                pzz = px.pz
            if(EID==3):
                print(pxx, pyy, pzz, np.sqrt(pxx**2+pyy**2+pzz**2))
        Q = []

        for PE in event.PEList:
            Q.append(PE.PMTId)
            EventID = event.TriggerNo
            ChannelID = PE.PMTId
            PETime =  PE.HitPosInWindow
            photonTime = PE.photonTime
            PulseTime = PE.PulseTime
            dETime = PE.dETime
            PEType = PE.PEType
            if(EID==3):
                #print(PulseTime)
                print(PEType)
                pass


# In[ ]:




