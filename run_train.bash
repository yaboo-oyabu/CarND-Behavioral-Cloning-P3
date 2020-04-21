#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO="udacity/carnd-term1-starter-kit"
COMMAND="/run.sh python clone.py"

docker run -it --rm -v ${DIR}:/src ${REPO} ${COMMAND}
