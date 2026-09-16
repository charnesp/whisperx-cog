#!/usr/bin/env bash
# Compute Docker image tags for the GHCR publish workflows (T7 + T4b, revue FIX).
#
# Entrées (env, fournies par GitHub Actions ou par les tests):
#   IMAGE           ghcr.io/owner/repo
#   GITHUB_SHA      commit sha (short8 extrait)
#   GITHUB_REF      refs/heads/main | refs/heads/master | refs/heads/feat/...
#   GITHUB_REF_NAME nom court de branche
#
# Sortie: lignes KEY=VALUE sur stdout (à `>> "$GITHUB_ENV"` dans le workflow):
#   IMAGE, TAGS, IS_DEFAULT_BRANCH, SHORT_SHA
#
# Garde :latest — égalité EXACTE sur refs/heads/main ou refs/heads/master.
# Jamais de prefix-match: 'refs/heads/m' matchait mainx/masterx/maint/main2
# et poussait :latest hors main (rejet revue T7+T4b, finding HIGH 1).
set -euo pipefail

IMAGE="${IMAGE:?IMAGE must be set (ghcr.io/owner/repo)}"
SHORT_SHA="${GITHUB_SHA::8}"

IS_DEFAULT_BRANCH=false
if [ "$GITHUB_REF" = "refs/heads/main" ] || [ "$GITHUB_REF" = "refs/heads/master" ]; then
  IS_DEFAULT_BRANCH=true
fi

TAGS="$IMAGE:sha-$SHORT_SHA $IMAGE:canary"
if [ "${GITHUB_REF_NAME:-}" = "feat/qwen3-asr-backend" ]; then
  TAGS="$TAGS $IMAGE:feat-qwen3-asr-backend"
fi
if [ "$IS_DEFAULT_BRANCH" = "true" ]; then
  TAGS="$TAGS $IMAGE:latest"
fi

echo "IMAGE=$IMAGE"
echo "TAGS=$TAGS"
echo "IS_DEFAULT_BRANCH=$IS_DEFAULT_BRANCH"
echo "SHORT_SHA=$SHORT_SHA"