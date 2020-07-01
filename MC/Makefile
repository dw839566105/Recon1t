fileNo:=$(shell echo {1..500})
dst:=$(fileNo:%=file%.h5)
all:$(dst)

file%.h5:
	python3 MCrecon.py $* /mnt/stage/douwei/Simulation/MC/ $@

