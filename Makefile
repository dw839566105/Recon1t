#program:=ML_time
#program:=ML_time_fixtaud
#program:=ML_notime
#program:=ML_notime_nocons
#program:=ML_weighted_average
#program:=Sph_corr
#program:=Sph_noc

dl:=/mnt/neutrino/02_PreAnalysis/run00000900

-include rl.mk

.PHONY: all

nl:=$($(notdir $(dl)))

all: $(nl:%=/mnt/stage/recon/900/%/sph.h5)

/mnt/stage/recon/900/%/sph.h5: $(dl)/PreAnalysis_Run900_File%.root
	mkdir -p $(dir $@)
	time python3 Recon_Sph_noc.py $^ $@ > $@.log 2>&1

rl.mk:
	echo "$(notdir $(dl)):=$(shell ./fl.sh $(dl))" > rl.mk

# Delete partial files when the processes are killed.
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:

