#!/bin/bash
set -e -o -x
SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"
TARGET_FOLDER="/datadrive/ARM"
SOURCE_ROOT=$(realpath $SCRIPT_DIR/../../../../)

while getopts f: parameter_Option
do case "${parameter_Option}"
in
f) TARGET_FOLDER=${OPTARG};;
esac
done

YOCTO_IMAGE="arm-yocto_imx-4.14"
cd $SCRIPT_DIR/docker
docker build -t $YOCTO_IMAGE -f Dockerfile.arm_yocto .

if [ ! -f $TARGET_FOLDER/bin/repo ]; then
    mkdir $TARGET_FOLDER/bin
    curl https://storage.googleapis.com/git-repo-downloads/repo > $TARGET_FOLDER/bin/repo
    chmod a+x $TARGET_FOLDER/bin/repo
fi

if [ ! -d $TARGET_FOLDER/fsl-arm-yocto-bsp ]; then
    mkdir $TARGET_FOLDER/fsl-arm-yocto-bsp
    cd $TARGET_FOLDER/fsl-arm-yocto-bsp
    $TARGET_FOLDER/bin/repo init -u https://source.codeaurora.org/external/imx/imx-manifest -b imx-linux-sumo -m imx-4.14.98-2.0.0_machinelearning.xml
    $TARGET_FOLDER/bin/repo sync
fi

YOCTO_CONTAINER="arm_yocto"
docker rm -f $YOCTO_CONTAINER || true
docker run --name $YOCTO_CONTAINER --volume $TARGET_FOLDER/fsl-arm-yocto-bsp:/fsl-arm-yocto-bsp --volume $SOURCE_ROOT:/onnxruntime_src $YOCTO_IMAGE /bin/bash /onnxruntime_src/tools/ci_build/github/linux/yocto_build_toolchain.sh &

wait $!

EXIT_CODE=$?

set -e
exit $EXIT_CODE



