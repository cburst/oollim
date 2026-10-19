#!/bin/sh
# Lightweight Platypus stub; all payload/runtime files are in the home folder.
set -eu
launcher="$HOME/oollim/launcher/launch.command"
if [ ! -f "$launcher" ]; then
  /usr/bin/osascript -e 'display alert "oollim is not installed" message "Run the Install oollim app first."'
  exit 1
fi
/usr/bin/open -a Terminal "$launcher"
