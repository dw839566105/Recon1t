.PHONY: all

order:=100
E:=point_axis
prog:=Recon_1t_charge_template
path:=/mnt/stage/douwei/Simulation/1t_root/$(E)/
save:=result_1t_$(E)_$(prog)
#sim:=$(wildcard $(path)1t_*.root)
sim:=$(wildcard $(path)1t_+?.??0_[z].root)
dst:=$(sim:$(path)%.root=$(save)/%.h5)
all:$(dst)

$(save)/%.h5:$(path)%.root
	mkdir -p $(save)
	python3 $(prog).py $^ $@ $(order) > log/$*.log

# Delete partial files when the processes are killed.
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:
