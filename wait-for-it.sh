#!/usr/bin/env bash
# wait-for-it.sh

set -e

host="$1"
shift
cmd="$@"

until pg_isready -h "$host" -p 5432; do
  >&2 echo "Postgres bekleniyor - $host"
  sleep 1
done

>&2 echo "Postgres hazır - başlatılıyor..."
exec $cmd
