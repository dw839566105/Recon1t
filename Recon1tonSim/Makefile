.PHONY: all

order:=10
E:=2.0MeV_dns
prog:=Recon_1t
path:=/mnt/stage/douwei/Simulation/1t_root/$(E)/
save:=result_1t_$(E)_$(prog)_2
#sim:=$(wildcard $(path)1t_*.root)
sim:=$(wildcard $(path)1t_??.??0_?.root)
dst:=$(sim:$(path)%.root=$(save)/%.h5)
all:$(dst)

$(save)/%.h5:$(path)%.root
	mkdir -p $(save)
	python3.6 $(prog).py $^ $@ $(order) > log/$*.log

# Delete partial files when the processes are killed.
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:
