#!/bin/bash

set -euo pipefail

usage() {
  printf 'Usage: %s {no_trim|trim|both}\n' "$0"
}

submit_variant() {
  local variant="$1"
  local preprocess_config
  local train_config

  case "$variant" in
    no_trim)
      preprocess_config="src/preprocessing/configs/spt_pipeline_no_trim.yaml"
      train_config="src/training/configs/spt_tranad_no_trim.yaml"
      ;;
    trim)
      preprocess_config="src/preprocessing/configs/spt_pipeline_trim.yaml"
      train_config="src/training/configs/spt_tranad_trim.yaml"
      ;;
    *)
      printf 'Unknown variant: %s\n' "$variant" >&2
      return 1
      ;;
  esac

  qsub -v "PREPROCESS_CONFIG_PATH=${preprocess_config},TRAIN_CONFIG_PATH=${train_config}" scripts/spt/swing_spt_example.pbs
}

if [ "$#" -ne 1 ]; then
  usage >&2
  exit 1
fi

case "$1" in
  no_trim)
    submit_variant no_trim
    ;;
  trim)
    submit_variant trim
    ;;
  both)
    submit_variant no_trim
    submit_variant trim
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 1
    ;;
 esac
