#!/bin/bash

# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD 3-Clause

# Loading the driver with gem5's bare bones ACPI implementation causes ACPI to
# be disabled, causing modprobe to fail. In particular, it fails on WMI init:
# https://elixir.bootlin.com/linux/v6.10.8/source/drivers/platform/x86/wmi.c#L1356
#
# To fix this we manually insmod all of the dependencies of the amdgpu module
# and a bandaid module containing missing ACPI symbols. The missing symbols are
# not important for gem5 so this does not cause any problems.
insmod /home/gem5/gem5_amdgpu_acpi.ko

modprobe drm_display_helper
modprobe drm_suballoc_helper
modprobe drm_exec
modprobe amddrm_buddy
modprobe amddrm_ttm_helper
modprobe amdkcl
modprobe amd-sched
modprobe amdttm
modprobe amdxcp
modprobe i2c_algo_bit

insmod /lib/modules/`uname -r`/updates/dkms/amdgpu.ko.zst ip_block_mask=0x6f ppfeaturemask=0 dpm=0 audio=0 ras_enable=0
