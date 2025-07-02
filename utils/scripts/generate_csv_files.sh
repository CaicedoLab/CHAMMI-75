#!/bin/bash

zips_file="$1"  # Get the first command-line argument (zips.txt)

if [ -z "$zips_file" ]; then
  echo "Usage: $0 <zips_file>"
  exit 1
fi

parallel -j 20 --colsep : --no-notice 'python ./generate_tile_hists.py -p {} -o /mnt/cephfs/mir/jcaicedo/morphem/dataset/sampling/content_filtering/v2 -c ~/FoundationModels/utils/scripts/thresholds.json --csv -l' :::: "$zips_file"

