#!/bin/sh

set -eu

DB_WAIT_HOST="${DB_HOST:-${DB_DEV_HOST:-db}}"
DB_WAIT_PORT="${DB_PORT:-${DB_DEV_PORT:-5432}}"

echo "Waiting for postgres at ${DB_WAIT_HOST}:${DB_WAIT_PORT}..."

while ! nc -z "$DB_WAIT_HOST" "$DB_WAIT_PORT"; do
  sleep 0.5
done

echo "PostgreSQL started"

if [ "${RUN_DB_MIGRATIONS:-true}" = "true" ]; then
  python manage.py migrate
fi

if [ "${RUN_SEED_ADMIN_USER:-false}" = "true" ]; then
  python manage.py seed_admin_user
fi

if [ "${RUN_COLLECTSTATIC:-true}" = "true" ]; then
  python manage.py collectstatic --no-input --clear
fi

exec "$@"
