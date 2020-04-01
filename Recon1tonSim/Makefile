.PHONY: all

path:=/mnt/stage/douwei/Simulation/5kt_root/2MeV/
save:=result
sim:=$(wildcard $(path)*.root)
dst:=$(sim:$(path)%.root=$(save)/%.h5)
all:$(dst)

$(save)/%.h5:$(path)%.root
	mkdir -p $(save)
	python3 Recon.py $^ $@ > $@.log

# Delete partial files when the processes are killed.
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:
