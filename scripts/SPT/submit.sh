#qsub -v CONFIG=configs/test_adv.yaml swing_spt_example.pbs
#qsub -v CONFIG=configs/default_epochs200.yaml swing_spt_example.pbs
#qsub -v CONFIG=configs/fast_epochs200.yaml swing_spt_example.pbs
#qsub -v CONFIG=configs/slow_epochs200.yaml swing_spt_example.pbs
#qsub -v CONFIG=configs/slow_epochs400_lambda0p1.yaml swing_spt_example.pbs
#qsub -v CONFIG=configs/slow_epochs400_lambda1p0.yaml swing_spt_example.pbs
#qsub -v CONFIG=configs/slow_epochs400_lambda10.yaml swing_spt_example.pbs
qsub -v CONFIG=configs/NEW_PLOTS.yaml swing_spt_example.pbs
