.PHONY: all

E:=track_oil
# 2.0MeV_dns: no dark noise, all PMTs are the same
# 2.0MeV_dn: no dark noise, PMTs are different
# 2.0MeV_xyz: dark noise thereshold 1
# 2.0MeV_20000: dark noise threshold 25
path:=/mnt/stage/douwei/Simulation/1t_root/$(E)/

upper_o:=0.648
lower_o:=0.40
radius_o:=$(shell seq -f '%.3f' $(upper_o) -0.002 $(lower_o))

upper_i:=0.40
lower_i:=0.01
radius_i:=$(shell seq -f '%.3f' $(upper_i) -0.01 $(lower_i))

radius:=$(shell echo $(radius_i) $(radius_o))

save_pe:=coeff_pe_1t_$(E)_1
save_time:=coeff_time_1t_$(E)_Lasso_1

src_pe:=$(radius:%=$(save_pe)/file_%.h5)
src_time:=$(radius:%=$(save_time)/file_%.h5)

order:=25

all:coeff_pe
#all:coeff_pe coeff_time
$(save_pe)/file_%.h5:
	mkdir -p $(save_pe)
	python3 PE_calib/main_calib_photon.py $* $(path) $@ $(order) > logs/PE$*.log

$(save_time)/file_%.h5:
	mkdir -p $(save_time)
	python3 Time_calib/main_calib.py $* $(path) $@ $(order) > logs/Time$*.log

coeff_pe:$(src_pe)
	python3 PE_calib/Gather.py ./$(save_pe)/ $(upper_o) $(lower_o) $(lower_i) $(order)

coeff_time:$(src_time)
	python3 Time_calib/Gather.py ./$(save_time)/ $(upper_o) $(lower_o) $(lower_i) $(order)

.SECONDARY:
