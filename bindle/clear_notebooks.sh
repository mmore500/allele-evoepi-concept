#!/bin/bash

################################################################################
echo
echo "running clear_notebooks.sh"
echo "---------------------------------------------"
################################################################################

# fail on error
set -e

################################################################################
echo
echo "other initialization"
echo "--------------------"
################################################################################

# adapted from https://stackoverflow.com/a/24114056
script_dir="$(dirname -- "$BASH_SOURCE")"
echo "script_dir ${script_dir}"

################################################################################
echo
echo "clear generated notebooks in current directory"
echo "----------------------------------------------"
################################################################################

# marimo notebooks are source .py files; the ipynb equivalents are generated
# artifacts created by execute_notebooks.sh, so remove any that exist.
shopt -s nullglob

for notebook in "${script_dir}/"*.py; do
  name="$(basename "${notebook%.*}")"
  rm -f "${script_dir}/${name}.ipynb"
done

shopt -u nullglob

################################################################################
echo
echo "recurse to subdirectories"
echo "-------------------------"
################################################################################

shopt -s nullglob

for script in "${script_dir}/"*/clear_notebooks.sh; do
  "${script}"
done

shopt -u nullglob
