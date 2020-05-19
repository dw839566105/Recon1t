#### Program Introduction
+. This program is for Jinping neutrino experiment, mainly for event reconstruction. We use spherical harmonics to get a template of 1t prototype detector, the prior is MC simulation by JSAP(Jinping Simulation and Analysis Package) mainly developed by Ziyi Guo.

+. Calib is using the simulated root files to get the spherical harmonic coefficients, the idea is proposed by Benda Xu.

+. Recon1tonSim is to reconstructed with the calib data, to make sure our templates is self-consistent. However, some position are ambiguous, which is being checked whether it is a lack of detector or the templates.

+. Recon1tonReal is to reconstruted with the real data, pre-analysis by Ziyi Guo and Yiyang Wu. Waveform Analysis is not included in this program. It is based on Recon1tonSim, and the program need to be updated in following version.

#### Challenges
1. total reflection is annoyed, no experiments has been successful before
2. how to get a suitable order coefficient (using sparse regression)
3. use r, theta, phi as the vertex have a unnatural boundary when r=0
4. whether r=0.36m and r=0.50m can be distinguished 

