.PHONY: all

E:=2.0MeV
path:=/mnt/stage/douwei/Simulation/1t_root/$(E)_015/
radius:=$(shell seq -f '%+.2f' +0.60 -0.01 -0.60)
save_pe:=coeff_pe_1t_$(E)
save_time:=coeff_time_1t_$(E)
src_pe:=$(radius:%=$(save_pe)/file_%.h5)
src_time:=$(radius:%=$(save_time)/file_%.h5)

all:coeff_pe coeff_time

$(save_pe)/file_%.h5:
	mkdir -p $(save_pe)
	python3 PE_calib/main_calib.py $* $(path) $@

$(save_time)/file_%.h5:
	mkdir -p $(save_time)
	python3 Time_calib/main_calib.py $* $(path) $@

coeff_pe:$(src_pe)
	python3 PE_calib/Gather.py ./$(save_pe)/ +0.60 -0.60 35

coeff_time:$(src_time)
	python3 Time_calib/Gather.py ./$(save_time)/ +0.60 -0.60 35
