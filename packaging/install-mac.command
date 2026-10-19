#!/bin/sh
set -eu
base=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
case "$(/usr/bin/uname -m)" in
  arm64) ;;
  *) printf 'oollim requires an Apple Silicon Mac.\n'; exit 1 ;;
esac
if [ ! -d "$base/stage" ] || [ ! -d "$base/oollim.app" ]; then
  printf 'Keep Install oollim.command beside stage/ and oollim.app.\n'; exit 1
fi
mkdir -p "$HOME/oollim/launcher" "$HOME/Applications"
/usr/bin/ditto "$base/stage" "$HOME/oollim/launcher"
/usr/bin/ditto "$base/oollim.app" "$HOME/Applications/oollim.app"
printf 'Installed launcher in ~/Applications. Runtime and models use ~/oollim.\n'
/usr/bin/open "$HOME/Applications/oollim.app"
