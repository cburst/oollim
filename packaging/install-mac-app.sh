#!/bin/sh
# Platypus installer: payload is bundled beside this script in Resources.
set -eu
base=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
fail() { printf '\nInstallation failed: %s\n' "$1"; exit 1; }
[ "$(/usr/bin/uname -m)" = arm64 ] || fail 'An Apple Silicon Mac is required.'
[ -d "$base/stage" ] && [ -d "$base/oollim.app" ] || fail 'The installer payload is incomplete. Download a fresh installer.'
printf 'Installing oollim…\n\n'
mkdir -p "$HOME/oollim/launcher" "$HOME/Applications" || fail 'Could not create the application folders in your home directory.'
/usr/bin/ditto "$base/stage" "$HOME/oollim/launcher" || fail 'Could not copy the runtime. Check available disk space and folder permissions.'
/usr/bin/ditto "$base/oollim.app" "$HOME/Applications/oollim.app" || fail 'Could not install the application shortcut.'
printf 'Installed: ~/Applications/oollim.app\n\nOpening first-run setup in Terminal. Keep Terminal open while dependencies and speech models download. The evaluation window opens when setup finishes.\n\nYou can close this installer window.\n'
/usr/bin/open "$HOME/Applications/oollim.app" || fail 'Open oollim manually from your home Applications folder to finish setup.'
