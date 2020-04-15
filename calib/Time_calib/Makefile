.PHONY: all

#file:=/mnt/stage/douwei/Simulation/1t_root/1MeV/
radius:=$(shell seq -f '%+.2f' +0.60 -0.01 -0.61)
save:=coeff_time_1t_1.5MeV
src:=$(radius:%=$(save)/file_%.h5)

all:$(src)

$(save)/file_%.h5:
	mkdir -p $(save)
	python3 Time_calib/main_calib.py $* $@
