dl:=/mnt/neutrino/02_PreAnalysis

-include rl.mk

.PHONY: all

sel:=0900 1115 1498 1692 1725 1846

all: $(foreach r,$(sel),$(patsubst %,/mnt/stage/recon/$(r)/%/tau.h5,$(srl-$(r))))

define r-tpl
s:=$(shell echo $(1) | sed 's/^0*//')
/mnt/stage/recon/$(1)/%/tau.h5: $(dl)/run0000$(1)/PreAnalysis_Run$$(s)_File%.root
	mkdir -p $$(dir $$@)
	time python3 Recon_Tau.py $$^ $$@ > $$@.log 2>&1

endef

$(foreach r,$(rl),$(eval $(call r-tpl,$(r))))

rl.mk:
	./rl.sh $(dl) > rl.mk

# Delete partial files when the processes are killed.
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:
