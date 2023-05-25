#!/bin/sh
export LD_LIBRARY_PATH="$(cat /stdlib.txt):$LD_LIBRARY_PATH"
exec "$@"
