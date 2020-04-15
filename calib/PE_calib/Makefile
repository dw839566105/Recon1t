.PHONY: all

radius:=$(shell seq -f '%+.2f' +0.60 -0.01 -0.60)
save:=coeff_pe_1t_1.5MeV
src:=$(radius:%=$(save)/file_%.h5)

all:$(src)

$(save)/file_%.h5:
	mkdir -p $(save)
	python3 PE_calib/main_calib.py $* $@
