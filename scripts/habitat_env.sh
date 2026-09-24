# Source from the Habitat shell scripts after ROOT is set.
# Another machine must export both variables. This checkout falls back to the
# habitat-lab and interpreter already used on the current server.

if [[ -z "${HABITAT_LAB:-}" && -d /amax/daihaojie/DPed-VLN/habitat-lab ]]; then
  HABITAT_LAB=/amax/daihaojie/DPed-VLN/habitat-lab
fi
if [[ -z "${HABITAT_PYTHON:-}" && -x /home/w61/miniconda3/envs/dpedvln/bin/python ]]; then
  HABITAT_PYTHON=/home/w61/miniconda3/envs/dpedvln/bin/python
fi
if [[ -z "${HABITAT_LAB:-}" || ! -d "${HABITAT_LAB}" ]]; then
  echo "Set HABITAT_LAB to a habitat-lab 0.3.x checkout."
  exit 1
fi
if [[ -z "${HABITAT_PYTHON:-}" || ! -x "${HABITAT_PYTHON}" ]]; then
  echo "Set HABITAT_PYTHON to the interpreter that has habitat-sim 0.3.1."
  exit 1
fi
export HABITAT_LAB HABITAT_PYTHON
export PYTHONPATH="${ROOT}:${HABITAT_LAB}${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${HABITAT_PYTHON}"
