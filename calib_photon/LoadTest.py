import rootpy
import ROOT as root
import numpy as np
import matplotlib.pyplot as plt

#f = root.TFile("test.root")
f = root.TFile("/mnt/stage/douwei/Simulation/1t_root/track/1t_+0.600_z.root")
myTree = f.Get("SimTriggerInfo")
N = np.int(10000000)
x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)
cnt = 0
for entry in myTree:
     # Now you have acess to the leaves/branches of each entry in the tree, e.g.
    for a in entry:
        events = a.truthList
        for b in events:
            for c in b.trackList:
                #print(c.nSegmentId, c.nPrimaryId, c.nTrackId)
                event_x = []
                event_y = []
                event_z = []
                tmp = []
                x_tmp = 0
                y_tmp = 0
                z_tmp = 0
                for d in c.StepPoints:
                    #print(d.nProcessType, d.fX, d.fY, d.fZ,np.sqrt(d.fX**2 + d.fY**2 + d.fZ**2))
                    print(c.nSegmentId, c.nParentTrackId, c.nTrackId, c.nPrimaryId, d.fX, d.fY, d.fZ, \
                            np.sqrt(d.fX**2 + d.fY**2 + d.fZ**2))

                    if(cnt >= N):
                        break
                    else:
                        if not ((cnt+1)%100000):
                            print(cnt+1)
                        ra = np.sqrt(d.fX**2 + d.fY**2 + d.fZ**2)
                        tmp.append(ra)
                        r_tmp = np.sqrt(x_tmp**2 + y_tmp**2 + z_tmp**2) 
                        if((r_tmp > 649) and (r_tmp < 651) and (ra>651)):
                            x[cnt] = d.fX
                            y[cnt] = d.fY
                            z[cnt] = d.fZ
                            
                            event_x.append(d.fX)
                            event_y.append(d.fY)
                            event_z.append(d.fZ)
                            
                            cnt = cnt + 1
                        x_tmp = d.fX
                        y_tmp = d.fY
                        z_tmp = d.fZ

                if(len(event_x)>10):
                    event_x = np.array(event_x)
                    event_y = np.array(event_y)
                    event_z = np.array(event_z)
                    print(np.vstack((event_x, event_y, event_z)).T, \
                          np.sqrt(np.sum((np.vstack((event_x, event_y, event_z))**2).T, axis=1)))
                    print(np.histogram(np.cos(np.arccos(event_z/np.sqrt(event_x**2+event_y**2+event_z**2))),bins=100))
                    print(haha)
                if(cnt >= N):
                    break

            if(cnt >= N):
                break

        if(cnt >= N):
            break
