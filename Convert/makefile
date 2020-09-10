srcL:=$(wildcard /mnt/stage/douwei/Simulation/1t_root/shell/*.root)
srcL:=$(wildcard /home/douwei/JSAP/DetectorStructure/output/1t_calib/*.root)
dstL:=$(srcL:%.root=%.h5)

.PHONY: all
all: $(dstL)

%.h5: %.root
	python3 ConvertTruth.py $^ $@
