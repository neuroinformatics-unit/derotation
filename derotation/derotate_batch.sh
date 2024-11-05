#!/bin/bash

python  derotation/derotate_batch.py \
    /nfs/winstor/margrie/SimonWeiler/RawData/Invivo_imaging/3photon_rotation/shared/230802_CAA_1120182/ \
    /nfs/winstor/margrie/SimonWeiler/RawData/Invivo_imaging/3photon_rotation/shared/230802_CAA_1120182/imaging/hold/rotation_00001.tif \
    /nfs/winstor/margrie/SimonWeiler/RawData/Invivo_imaging/3photon_rotation/shared/230802_CAA_1120182/aux_stim/230802_CAA_1120182_rotation_1_001.bin \
    /ceph/margrie/laura/test/
