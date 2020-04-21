#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TARGET="2020-04-21"
REPO="udacity/carnd-term1-starter-kit"
COMMAND="/run.sh python drive.py output/${TARGET}/model.h5 output/${TARGET}/runl"

docker run -it --rm -p 4567:4567 -v ${DIR}:/src ${REPO} ${COMMAND}
