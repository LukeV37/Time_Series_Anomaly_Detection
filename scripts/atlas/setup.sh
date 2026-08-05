#!/bin/bash
# Build a virtual environment for Time_Series_Anomaly_Detection pbeast fetching.
# Run from anywhere: ./scripts/ATLAS/setup.sh
# Then activate with: source scripts/ATLAS/activate_atom.sh
# Override the env path with: export PBEAST_VENV_DIR=/path/to/venv

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

if [ -f "$REPO_ROOT/export.sh" ]; then
  source "$REPO_ROOT/export.sh"
fi

: "${PBEAST_VENV_DIR:=$REPO_ROOT/venv_time_series_anomaly_detection}"
: "${TDAQ_RELEASE:=tdaq-12-00-00}"

echo -e "${GREEN}Setting up Time_Series_Anomaly_Detection environment (${PBEAST_VENV_DIR})...${NC}"

echo -e "${YELLOW}Sourcing TDAQ release...${NC}"
source /cvmfs/atlas.cern.ch/repo/sw/tdaq/tools/cmake_tdaq/bin/cm_setup.sh "$TDAQ_RELEASE"

export PYTHONNOUSERSITE=1

# pandas HDF5 I/O needs the LCG tables/h5py runtime libs.
export LD_LIBRARY_PATH="/cvmfs/sft.cern.ch/lcg/releases/LCG_106b/hdf5/1.14.3/${CMTCONFIG}/lib:/cvmfs/sft.cern.ch/lcg/releases/LCG_106b/blosc2/2.5.1/${CMTCONFIG}/lib64:/cvmfs/sft.cern.ch/lcg/releases/LCG_106b/blosc/1.11.1/${CMTCONFIG}/lib64:${LD_LIBRARY_PATH}"

[ -d "$PBEAST_VENV_DIR" ] && rm -rf "$PBEAST_VENV_DIR"
python3 -m venv "$PBEAST_VENV_DIR"
source "$PBEAST_VENV_DIR/bin/activate"

python3 -m pip install --upgrade pip setuptools wheel

# Pin packages to versions that stay compatible with the TDAQ stack.
python3 -m pip install -r "$SCRIPT_DIR/requirements-pbeast.txt"

# Make the repo's src/ importable from the venv.
cat > "$PBEAST_VENV_DIR/lib/python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/time_series_anomaly_detection_src.pth" <<EOF
$REPO_ROOT/src
EOF

echo -e "${GREEN}Setup complete.${NC}"
echo -e "${YELLOW}Activate with: source $REPO_ROOT/scripts/ATLAS/activate_atom.sh${NC}"
