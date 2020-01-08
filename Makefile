#program:=ML_time
#program:=ML_time_fixtaud
#program:=ML_notime
#program:=ML_notime_nocons
#program:=ML_weighted_average
#program:=Sph_corr
#program:=Sph_noc

.PHONY: all
filepath:=/srv/JinpingData/Jinping_1ton_Data/02_PreAnalysis/run00000900/PreAnalysis_Run900_File0.root
all:$(filepath)
	echo 'python3 Recon_$(program).py $(filepath) ./test_notime_nocons.h5 > notime_nocons.log'
