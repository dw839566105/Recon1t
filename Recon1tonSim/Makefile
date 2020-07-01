.PHONY: all

order:=15
E:=2.0MeV_015
path:=/mnt/stage/douwei/Simulation/1t_root/$(E)/
save:=result_1t_$(E)_15_MC2
#sim:=$(wildcard $(path)1t_*.root)
sim:=$(wildcard $(path)1t_*.root)
dst:=$(sim:$(path)%.root=$(save)/%.h5)
all:$(dst)

$(save)/%.h5:$(path)%.root
	mkdir -p $(save)
	python3 Recon_1t_cubic.py $^ $@ $(order) > log/$*.log

# Delete partial files when the processes are killed.
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:
