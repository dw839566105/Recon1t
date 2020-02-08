.PHONY: all

files:=$(wildcard ../Convert/test*.h5)
target:=$(files:../Convert/test%.h5=recon%.h5)

all:$(target),mid

mid: 
	cd ../Convert && make

recon%.h5: ../Convert/test%.h5
	python3 Recon_Tau.py $^ $@ > $^.log




