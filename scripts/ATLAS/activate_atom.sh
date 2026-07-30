#!/bin/bash
# Activate the Time_Series_Anomaly_Detection environment with the TDAQ + LCG runtime.
# Usage: source activate_atom.sh   (run from the Time_Series_Anomaly_Detection/ directory)

if [ -f .env ]; then
  source ./.env
fi

: "${VENV_DIR:=venv_time_series_anomaly_detection}"
: "${TDAQ_RELEASE:=tdaq-12-00-00}"

source /cvmfs/atlas.cern.ch/repo/sw/tdaq/tools/cmake_tdaq/bin/cm_setup.sh "$TDAQ_RELEASE"

export PYTHONNOUSERSITE=1

export LD_LIBRARY_PATH="/cvmfs/sft.cern.ch/lcg/releases/LCG_106b/hdf5/1.14.3/${CMTCONFIG}/lib:/cvmfs/sft.cern.ch/lcg/releases/LCG_106b/blosc2/2.5.1/${CMTCONFIG}/lib64:/cvmfs/sft.cern.ch/lcg/releases/LCG_106b/blosc/1.11.1/${CMTCONFIG}/lib64:${LD_LIBRARY_PATH}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"

source "$VENV_DIR/bin/activate"

echo "TDAQ sourced; ${VENV_DIR} active; user-site disabled"
echo "DATA_PATH=${DATA_PATH:-$(pwd)/src/pbeast_fetcher/data}"
