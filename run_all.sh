#!/bin/sh
# Runs every step of the video in order. About 6 minutes on a laptop.
set -e
cd "$(dirname "$0")"
for f in step1_digits step2_deeper step3_do_nothing step4_plus step5_signal step6_delete exercise tests; do
  echo "== $f =="
  python3 $f.py
  echo
done
