#!/bin/bash

# Example script to run batch alignment on CASP15 targets
# This script demonstrates how to use the batch_align_casp15.py script

cd /store/deepfold/users/baehanjin/work/OTalign

source .venv/bin/activate

python scripts/batch_align_casp.py \
    casp15_jobs/omit_2.txt \
    --casp-base /gpfs/deepfold/users/baehanjin/work/casp15 \
    --model ProtT5_XL_UniRef50 \
    --device cuda:0\
    --enable-filtering \
    --filter-threshold 0.5 \
    --batch-size 1 \
    --log-file casp15_jobs/logs/casp15_batch_alignment_omit_2.log

echo "Batch alignment completed. Check casp15_jobs/logs/casp15_batch_alignment_omit_2.log for details."
