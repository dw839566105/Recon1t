SHELL:=/bin/bash
dl:=/mnt/neutrino/02_PreAnalysis
rd:=/mnt/stage/recon

-include rl.mk

.PHONY: all

sel:=0900 1115 1498 1567 1569 1571 1680 1682 1684 1692 1706 1718 1725 1773 1840 1842 1846

all: $(sel:%=$(rd)/%/es.pdf)

define r-tpl
s:=$(shell echo $(1) | sed 's/^0*//')
$(rd)/$(1)/%/tau.h5: $(dl)/run0000$(1)/PreAnalysis_Run$$(s)_File%.root
	mkdir -p $$(dir $$@)
	time python3 Recon_Tau.py $$^ -o $$@ > $$@.log 2>&1

$(rd)/$(1)/es.pdf: $(srl-$(1):%=$(rd)/$(1)/%/tau.h5)
	python3 es.py -o $$@ $$^

endef

$(foreach r,$(rl),$(eval $(call r-tpl,$(r))))

rl.mk:
	./rl.sh $(dl) > rl.mk

# Delete partial files when the processes are killed.
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:
