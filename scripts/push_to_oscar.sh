#!/usr/bin/env bash
# Push raw videos and AIST++ 2D keypoints from this laptop to Oscar.
#
# Fixes three problems with the naive `rsync -av data/... user@host:~/scratch/.../data/...`:
#   1) The remote parent directory does not exist yet, and Apple's rsync 2.6.9 does
#      not support --mkpath. We pre-create it over SSH.
#   2) Every SSH to Oscar requires Duo 2FA. Without multiplexing, you would be
#      Duo-prompted on every rsync/ssh. We open one ControlMaster socket and
#      reuse it for all subsequent connections => one Duo for the whole run.
#   3) The transfer is ~5.4 GB over SSH and may break. We use --partial so a
#      re-run resumes where it left off.
#
# Overrides (all optional):
#   OSCAR_USER         (default: mwang264)
#   OSCAR_HOST         (default: ssh.ccv.brown.edu)
#   OSCAR_REMOTE_ROOT  (default: ~/scratch/projects/CV_Tool_for_Dance_Choreography_Practice)
#
# Usage:
#   bash scripts/push_to_oscar.sh
set -euo pipefail

OSCAR_USER="${OSCAR_USER:-mwang264}"
OSCAR_HOST="${OSCAR_HOST:-ssh.ccv.brown.edu}"
OSCAR_REMOTE_ROOT="${OSCAR_REMOTE_ROOT:-~/scratch/projects/CV_Tool_for_Dance_Choreography_Practice}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VIDEOS_SRC="data/raw_videos"
KPS_SRC="data/labels/aistpp/keypoints2d_raw"

for src in "$VIDEOS_SRC" "$KPS_SRC"; do
  if [[ ! -d "$src" ]]; then
    echo "ERROR: local source directory missing: $src" >&2
    exit 1
  fi
done

# One shared ControlMaster socket => one Duo prompt for the whole script.
# macOS caps Unix-socket paths at 104 bytes, so we use ~/.ssh with the short
# %C hash (16 chars) instead of a long /var/folders tmpdir.
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
CTRL_PATH="$HOME/.ssh/cm-%C"
SSH_OPTS=(
  -o "ControlMaster=auto"
  -o "ControlPath=$CTRL_PATH"
  -o "ControlPersist=60m"
  -o "ServerAliveInterval=30"
  -o "ServerAliveCountMax=6"
)

cleanup() {
  ssh "${SSH_OPTS[@]}" -O exit "$OSCAR_USER@$OSCAR_HOST" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "==> Opening SSH master connection to $OSCAR_USER@$OSCAR_HOST"
echo "    (expect Brown login banner + ONE Duo prompt; later steps reuse this socket)"
ssh "${SSH_OPTS[@]}" "$OSCAR_USER@$OSCAR_HOST" \
  "echo 'connected as '\$(whoami)' on '\$(hostname)"

REMOTE_VIDEOS="$OSCAR_REMOTE_ROOT/data/raw_videos"
REMOTE_KPS="$OSCAR_REMOTE_ROOT/data/labels/aistpp/keypoints2d_raw"

echo "==> Creating remote directories on Oscar"
ssh "${SSH_OPTS[@]}" "$OSCAR_USER@$OSCAR_HOST" \
  "mkdir -p $REMOTE_VIDEOS $REMOTE_KPS && echo 'remote dirs ready'"

# -a  archive (recursive + preserve perms/times/links)
# -v  verbose
# --progress  per-file progress
# --partial   keep partially-transferred files so re-running resumes them
# -h          human-readable sizes
# --stats     summary at the end
# -e "ssh ..."  route rsync's ssh through the shared ControlMaster
RSYNC_SSH="ssh ${SSH_OPTS[*]}"
RSYNC_OPTS=(-avh --progress --partial --stats -e "$RSYNC_SSH")

echo "==> Transferring raw videos ($(du -sh "$VIDEOS_SRC" | awk '{print $1}'))"
rsync "${RSYNC_OPTS[@]}" "$VIDEOS_SRC/" \
  "$OSCAR_USER@$OSCAR_HOST:$REMOTE_VIDEOS/"

echo "==> Transferring AIST++ 2D keypoints ($(du -sh "$KPS_SRC" | awk '{print $1}'))"
rsync "${RSYNC_OPTS[@]}" "$KPS_SRC/" \
  "$OSCAR_USER@$OSCAR_HOST:$REMOTE_KPS/"

echo "==> Verifying remote file counts"
ssh "${SSH_OPTS[@]}" "$OSCAR_USER@$OSCAR_HOST" bash -s <<EOF
set -e
echo "videos:    \$(ls -1 $REMOTE_VIDEOS | wc -l) files, \$(du -sh $REMOTE_VIDEOS | awk '{print \$1}')"
echo "keypoints: \$(ls -1 $REMOTE_KPS    | wc -l) files, \$(du -sh $REMOTE_KPS    | awk '{print \$1}')"
EOF

echo "==> Done."
