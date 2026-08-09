#!/usr/bin/env bash
set -euo pipefail

NGSPICE_VERSION="46"
NGSPICE_ARCHIVE_SHA256="a0d1699af1940b06649276dcd6ff5a566c8c0cad01b2f7b5e99dedbb4d64c19b"
NGSPICE_URL="https://sourceforge.net/projects/ngspice/files/ng-spice-rework/46/ngspice-46.tar.gz/download"

if ngspice --version 2>&1 | grep -q "ngspice-${NGSPICE_VERSION}"; then
  echo "ngspice-${NGSPICE_VERSION} already present"
  exit 0
fi

run_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

run_root apt-get -o DPkg::Lock::Timeout=300 update -q
run_root apt-get -o DPkg::Lock::Timeout=300 install -y -q \
  build-essential ca-certificates curl libxaw7-dev libreadline-dev \
  bison flex libfftw3-dev

BUILD_ROOT="$(mktemp -d /tmp/phy-chip-env-ngspice46.XXXXXX)"
trap 'rm -rf "$BUILD_ROOT"' EXIT
ARCHIVE="$BUILD_ROOT/ngspice-46.tar.gz"

curl -fL --retry 3 --retry-all-errors --connect-timeout 30 \
  "$NGSPICE_URL" -o "$ARCHIVE"
printf '%s  %s\n' "$NGSPICE_ARCHIVE_SHA256" "$ARCHIVE" | sha256sum -c -
tar xzf "$ARCHIVE" -C "$BUILD_ROOT"

cd "$BUILD_ROOT/ngspice-${NGSPICE_VERSION}"
./configure --disable-debug --with-readline=yes --enable-openmp
make -j1 CCLD=g++
run_root make install CCLD=g++
run_root ldconfig

/usr/local/bin/ngspice --version 2>&1 | grep -q "ngspice-${NGSPICE_VERSION}"
echo "installed ngspice-${NGSPICE_VERSION}"
