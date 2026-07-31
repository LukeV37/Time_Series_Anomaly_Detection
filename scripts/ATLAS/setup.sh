#!/bin/bash
# Build a virtual environment for Time_Series_Anomaly_Detection pbeast fetching.
# Run from the repo root: ./setup.sh
# Then activate with:     source activate_atom.sh
# Override the env path with: export PBEAST_VENV_DIR=/path/to/venv

set -e

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

if [ -f .env ]; then
  source ./.env
fi

: "${PBEAST_VENV_DIR:=venv_time_series_anomaly_detection}"
: "${TDAQ_RELEASE:=tdaq-12-00-00}"

echo -e "${GREEN}Setting up Time_Series_Anomaly_Detection environment (${PBEAST_VENV_DIR})...${NC}"

echo -e "${YELLOW}Sourcing TDAQ release...${NC}"
source /cvmfs/atlas.cern.ch/repo/sw/tdaq/tools/cmake_tdaq/bin/cm_setup.sh "$TDAQ_RELEASE"

export PYTHONNOUSERSITE=1

# pandas HDF5 I/O needs the LCG tables/h5py runtime libs.
export LD_LIBRARY_PATH="/cvmfs/sft.cern.ch/lcg/releases/LCG_106b/hdf5/1.14.3/${CMTCONFIG}/lib:/cvmfs/sft.cern.ch/lcg/releases/LCG_106b/blosc2/2.5.1/${CMTCONFIG}/lib64:/cvmfs/sft.cern.ch/lcg/releases/LCG_106b/blosc/1.11.1/${CMTCONFIG}/lib64:${LD_LIBRARY_PATH}"

[ -d "$PBEAST_VENV_DIR" ] && rm -rf "$PBEAST_VENV_DIR"
python3 -m venv --system-site-packages "$PBEAST_VENV_DIR"
source "$PBEAST_VENV_DIR/bin/activate"

python3 -m pip install --upgrade pip setuptools wheel

# Pin numpy to the TDAQ-compatible 1.26 line; numpy 2.x breaks the release.
python3 -m pip install "numpy==1.26.4"

# Minimal Python deps for fetch_one_run.py and src/pbeast_fetcher.
python3 -m pip install pandas pyyaml python-dateutil pytz

# Make the repo's src/ importable from the venv.
cat > "$PBEAST_VENV_DIR/lib/python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/time_series_anomaly_detection_src.pth" <<EOF
$(pwd)/src
EOF

echo -e "${GREEN}Setup complete.${NC}"
echo -e "${YELLOW}Activate with: source activate_atom.sh${NC}"
