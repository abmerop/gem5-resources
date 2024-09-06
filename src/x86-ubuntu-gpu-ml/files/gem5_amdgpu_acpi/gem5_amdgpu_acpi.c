/*
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <acpi/video.h>

MODULE_AUTHOR("Matthew Poremba");
MODULE_DESCRIPTION("gem5 amdgpu ACPI symbol overrides");
MODULE_LICENSE("Dual BSD/GPL");

void acpi_video_register_backlight(void)
{
    printk(KERN_INFO "gem5 stub acpi_video_register_backlight called");
}
EXPORT_SYMBOL(acpi_video_register_backlight);

enum acpi_backlight_type __acpi_video_get_backlight_type(bool native, bool *auto_detect)
{
    // Unclear if this matters
    return acpi_backlight_none;
}
EXPORT_SYMBOL(__acpi_video_get_backlight_type);

static int __init gem5_amdgpu_acpi_init(void)
{
    printk(KERN_INFO "Loading gem5 video overrides");
    return 0;
}

static void __exit gem5_amdgpu_acpi_exit(void)
{
    printk(KERN_INFO "Unloading gem5 video overrides");
}

module_init(gem5_amdgpu_acpi_init);
module_exit(gem5_amdgpu_acpi_exit);
