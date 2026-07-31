#!/bin/bash
# Activate the Time_Series_Anomaly_Detection environment with the TDAQ + LCG runtime.
# Usage: source scripts/ATLAS/activate_atom.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -f "$REPO_ROOT/export.sh" ]; then
  source "$REPO_ROOT/export.sh"
fi

: "${PBEAST_VENV_DIR:=$REPO_ROOT/venv_time_series_anomaly_detection}"
: "${TDAQ_RELEASE:=tdaq-12-00-00}"

source /cvmfs/atlas.cern.ch/repo/sw/tdaq/tools/cmake_tdaq/bin/cm_setup.sh "$TDAQ_RELEASE"

export PYTHONNOUSERSITE=1

export LD_LIBRARY_PATH="/cvmfs/sft.cern.ch/lcg/releases/LCG_106b/hdf5/1.14.3/${CMTCONFIG}/lib:/cvmfs/sft.cern.ch/lcg/releases/LCG_106b/blosc2/2.5.1/${CMTCONFIG}/lib64:/cvmfs/sft.cern.ch/lcg/releases/LCG_106b/blosc/1.11.1/${CMTCONFIG}/lib64:${LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:${PYTHONPATH}}"

source "$PBEAST_VENV_DIR/bin/activate"

echo "TDAQ sourced; ${PBEAST_VENV_DIR} active; user-site disabled"
