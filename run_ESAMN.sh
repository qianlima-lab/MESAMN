#!/bin/bash

nohup python ./ESAMN_TS.py --cuda_device 0 --outfile ESAMN_MTS.csv >ESAMN.out 2>&1 &
nohup python ./UTD.py >ESAMN_UTD.out 2>&1 &
nohup python ./HDM05.py >ESAMN_HDM05.out 2>&1 &
nohup python ./F3D_baseline.py >ESAMN_F3D.out 2>&1 &