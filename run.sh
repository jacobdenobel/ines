#!/usr/bin/env bash
set -euo pipefail

DIMS="2,3,5,10,20,40,100"

KINDS=(
    sphere
    ellipse
    discus
    cigar
)

# label | center | statistic
VARIANTS=(
  "INES best best"
  "INES-MP best weighted"
  "Round round weighted"
  "SRound sround weighted"
  "Med-W weighted_median weighted"
  "Disc-W weighted_discrete weighted"
  "Disc-U discrete weighted"
)

for kind in "${KINDS[@]}"; do
  echo
  echo "============================================================"
  echo "Benchmark: ${kind}"
  echo "============================================================"

  for variant in "${VARIANTS[@]}"; do
    read -r label center statistic <<< "$variant"

    echo
    echo "------------------------------------------------------------"
    echo "Variant: ${label} | center=${center} | statistic=${statistic}"
    echo "------------------------------------------------------------"

    ines benchmark \
      --kind "$kind" \
      --dims "$DIMS" \
      --statistic "$statistic" \
      --center "$center" & 
  done
done