srcL:=$(wildcard /mnt/stage/douwei/Simulation/1t_root/2.0MeV_xyz/*.root)
dstL:=$(srcL:%.root=%.h5)

.PHONY: all
all: $(dstL)

%.h5: %.root
	python3 ConvertTruth.py $^ $@
