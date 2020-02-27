.PHONY: all

E:=1.8MeV
radius:=$(shell seq -f '%+.2f' -0.60 0.01 0.60)
		
src:=$(radius:%=coeff/$(E)/file_%.h5)

all:$(src)

coeff/$(E)/file_%.h5:
	mkdir -p coeff
	mkdir -p coeff/$(E)
	python3 main_calib.py $(E) $* $@
