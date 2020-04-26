.PHONY: all

E:=2.0MeV
path:=/mnt/stage/douwei/Simulation/1t_root/$(E)_T/
upper:=+0.65
lower:=-0.65
radius:=$(shell seq -f '%+.2f' $(upper) -0.01 $(lower))
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
	python3 PE_calib/Gather.py ./$(save_pe)/ $(upper) $(lower) 35

coeff_time:$(src_time)
	python3 Time_calib/Gather.py ./$(save_time)/ $(upper) $(lower) 35
