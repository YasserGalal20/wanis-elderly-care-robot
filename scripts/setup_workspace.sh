#!/usr/bin/env bash
# Reconstruct the full Wanis workspace: clone the upstream packages at the
# exact commits we built against, then apply our patches on top.
set -euo pipefail

WS="${1:-$HOME/wanis_ws}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> workspace: $WS"
mkdir -p "$WS/src" "$WS/firmware"

command -v vcs >/dev/null || { echo "vcstool missing: sudo apt install python3-vcstool"; exit 1; }

echo "==> cloning upstream dependencies (pinned)"
vcs import "$WS" < "$HERE/robot.repos"

echo "==> applying our patches"
for p in "$HERE"/patches/*.patch; do
  name="$(basename "$p" .patch)"
  for target in "$WS/src/$name" "$WS/firmware/$name"; do
    [ -d "$target" ] || continue
    echo "    $name"
    git -C "$target" apply --3way "$p" || echo "    !! $name did not apply cleanly - resolve manually"
  done
done

echo "==> linking our own packages"
ln -sfn "$HERE/src/person_follower" "$WS/src/person_follower"
ln -sfn "$HERE/src/wanis_bringup"   "$WS/src/wanis_bringup"

echo "==> done. Next:"
echo "    bash $HERE/scripts/download_models.sh"
echo "    cd $WS && colcon build --symlink-install"
