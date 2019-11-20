#!/bin/bash
set -e

cd /fsl-arm-yocto-bsp
LOCAL_CFG="/fsl-arm-yocto-bsp/buildxwayland/conf/local.conf"
if [ -f $LOCAL_CFG ]; then
    rm -rf $LOCAL_CFG
fi

EULA=1 MACHINE=imx8qmmek DISTRO=fsl-imx-xwayland source ./fsl-setup-release.sh -b buildxwayland
echo "BBLAYERS += \" \${BSPDIR}/sources/meta-imx-machinelearning \"" >> conf/bblayers.conf

cat >> $LOCAL_CFG <<EOT
EXTRA_IMAGE_FEATURES = " dev-pkgs debug-tweaks tools-debug tools-sdk ssh-server-openssh"

IMAGE_INSTALL_append = " net-tools iputils dhcpcd"

IMAGE_INSTALL_append = " which gzip python python-pip"
IMAGE_INSTALL_append = " wget cmake gtest git zlib patchelf"
IMAGE_INSTALL_append = " nano grep vim tmux swig tar unzip"
IMAGE_INSTALL_append = " parted e2fsprogs e2fsprogs-resize2fs"

IMAGE_INSTALL_append = " opencv python-opencv"
PACKAGECONFIG_remove_pn-opencv_mx8 = "python3"
PACKAGECONFIG_append_pn-opencv_mx8 = " dnn python2 qt5 jasper openmp test neon"

PACKAGECONFIG_remove_pn-opencv_mx8 = "opencl"
PACKAGECONFIG_remove_pn-arm-compute-library = "opencl"

TOOLCHAIN_HOST_TASK_append = " nativesdk-cmake nativesdk-make"

IMAGE_INSTALL_append = " arm-compute-library armnn"
PREFERRED_VERSION_opencv = "4.0.1%"
EOT

bitbake fsl-image-qt5 -c populate_sdk