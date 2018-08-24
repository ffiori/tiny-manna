#!/bin/bash

pushd () {
    command pushd "$@" > /dev/null
}

popd () {
    command popd "$@" > /dev/null
}

TMPGCC="$(mktemp -d)"

pushd "$TMPGCC"
for cmd in /usr/bin/x86_64-linux-gnu-*-7; do
  ln -s "$cmd" "$(echo $cmd|sed 's#/usr/bin/x86_64-linux-gnu-\(.*\)-7#\1#')"
done
popd

export PATH="$TMPGCC:$PATH"

#~ exec /bin/bash

# hacerle source para ejecutar nvcc 9.2 con gcc 7.3
