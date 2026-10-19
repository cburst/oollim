#!/bin/sh
set -eu
base=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
unset PYTHONHOME PYTHONPATH
"$base/python/bin/python3.11" "$base/bootstrap.py" || { printf '\nPress Return to close...'; read -r answer; }
