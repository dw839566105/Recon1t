srcL:=$(wildcard /mnt/stage/douwei/Simulation/5kt_root/1MeV/*.root)
dstL:=$(srcL:%.root=%.h5)

.PHONY: all
all: $(dstL)

%.h5: %.root
	python3 ConvertTruth.py $^ $@
