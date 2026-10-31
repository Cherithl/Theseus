#!/usr/bin/env bash
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$RECIPE_DIR/recipe-environment.bash"
recipe_parse_arguments "$@"
CASES="$THESEUS_TEST_OUTPUT/cases"

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "prepare" ]]; then
  echo "==> Prepare HLL/Roe numerical-flux inputs"
  # THESEUS_SHOW prepare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/tests/numflux_regression_test.py" prepare \
    --source "$THESEUS_SOURCE" --results "$CASES" --database "$THESEUS_PREFIX/database" \
    --build "$THESEUS_BUILD_STANDARD"
  # THESEUS_SHOW_END
fi

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "run" ]]; then
  echo "==> Run 20-step HLL/Roe cases"
  cd "$THESEUS_SOURCE"
  # THESEUS_SHOW run
  for case in cpg-hll-cavity cpg-roe-cavity lte-hll-tgv; do
    echo "Running $case ($THESEUS_RANKS ranks, $THESEUS_DEVICE); log: $CASES/$case/run.log"
    scripts/run_theseus.sh -e "$THESEUS_BUILD_STANDARD/theseus" \
      -c "$CASES/$case/config.json" -o "$CASES/$case/launch" \
      -p "$THESEUS_RANKS" -r "$THESEUS_DEVICE" -P -R -k \
      > "$CASES/$case/run.log" 2>&1
  done
  # THESEUS_SHOW_END
fi

if [[ "$THESEUS_RECIPE_STAGE" == "all" || "$THESEUS_RECIPE_STAGE" == "compare" ]]; then
  echo "==> Compare fields against CPU main for each numerical flux"
  # THESEUS_SHOW compare
  "$THESEUS_PYTHON" "$THESEUS_SOURCE/tests/numflux_regression_test.py" check \
    --source "$THESEUS_SOURCE" --results "$CASES" --device "$THESEUS_DEVICE"
  # THESEUS_SHOW_END
fi
