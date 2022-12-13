#!/bin/bash

SRC="data/scannet"
MASK_PATH="data/scannet/mask_border_10px.png"

for scene in ${SRC}/* ; do
    if [ -d "${scene}" ]; then
        echo "cp ${MASK_PATH} ${scene}"
        cp ${MASK_PATH} ${scene}
    fi
done
