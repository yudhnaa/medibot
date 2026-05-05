#!/usr/bin/env sh
set -eu

R2_LIBS_DEST="${R2_LIBS_DEST:-nlp/services/libs}"
R2_LIBS_PREFIX="${R2_LIBS_PREFIX:-libs}"

if [ -z "${R2_ACCOUNT_ID:-}" ]; then
  echo "R2_ACCOUNT_ID is required" >&2
  exit 1
fi

if [ -z "${R2_BUCKET:-}" ]; then
  echo "R2_BUCKET is required" >&2
  exit 1
fi

if [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
  echo "AWS_ACCESS_KEY_ID is required" >&2
  exit 1
fi

if [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
  echo "AWS_SECRET_ACCESS_KEY is required" >&2
  exit 1
fi

if [ "$R2_LIBS_PREFIX" = "/" ] || [ "$R2_LIBS_PREFIX" = "." ]; then
  echo "R2_LIBS_PREFIX must not point at the bucket root" >&2
  exit 1
fi

endpoint_url="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
source_uri="s3://${R2_BUCKET}/${R2_LIBS_PREFIX%/}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-auto}"

mkdir -p "$R2_LIBS_DEST"

sync_args="s3 sync ${source_uri} ${R2_LIBS_DEST} --endpoint-url ${endpoint_url} --delete --no-progress"

if command -v aws >/dev/null 2>&1; then
  aws $sync_args
elif command -v docker >/dev/null 2>&1; then
  docker run --rm \
    -e AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY \
    -e AWS_DEFAULT_REGION \
    -v "$(pwd)":/work \
    -w /work \
    amazon/aws-cli:2.17.35 $sync_args
else
  echo "Install awscli or Docker before syncing R2 libs" >&2
  exit 1
fi
