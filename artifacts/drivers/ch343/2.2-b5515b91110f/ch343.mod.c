#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/elfnote-lto.h>
#include <linux/export-internal.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

#ifdef CONFIG_UNWINDER_ORC
#include <asm/orc_header.h>
ORC_HEADER;
#endif

BUILD_SALT;
BUILD_LTO_INFO;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_MITIGATION_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif



static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0xd80d68d5, "usb_deregister" },
	{ 0x8e17b3ae, "idr_destroy" },
	{ 0x54b1fac6, "__ubsan_handle_load_invalid_value" },
	{ 0x4c03a563, "random_kmalloc_seed" },
	{ 0xf5c9f2db, "kmalloc_caches" },
	{ 0x94cba22d, "kmalloc_trace" },
	{ 0xcd9c13a3, "tty_termios_hw_change" },
	{ 0xbd394d8, "tty_termios_baud_rate" },
	{ 0x71b5c9ab, "tty_encode_baud_rate" },
	{ 0xf0fdf6cb, "__stack_chk_fail" },
	{ 0x9be9f375, "tty_port_open" },
	{ 0x56d23fc5, "usb_ifnum_to_if" },
	{ 0xc82c9007, "usb_get_intf" },
	{ 0xb8f11603, "idr_alloc" },
	{ 0xd9a5ea54, "__init_waitqueue_head" },
	{ 0xcefb0c9f, "__mutex_init" },
	{ 0x18f8da7, "tty_port_init" },
	{ 0x4b3c93ab, "usb_alloc_coherent" },
	{ 0xffa75ead, "usb_alloc_urb" },
	{ 0xb165f690, "_dev_info" },
	{ 0x9022ddfa, "usb_register_dev" },
	{ 0x84f4d8ae, "usb_driver_claim_interface" },
	{ 0x432b32ba, "tty_port_register_device" },
	{ 0xf6daa2cc, "usb_driver_release_interface" },
	{ 0x4f0b66ca, "usb_free_urb" },
	{ 0xaf85897d, "usb_free_coherent" },
	{ 0x31987b81, "__tty_insert_flip_string_flags" },
	{ 0x20b9e069, "tty_flip_buffer_push" },
	{ 0x20978fb9, "idr_find" },
	{ 0x296695f, "refcount_warn_saturate" },
	{ 0xd34a8420, "seq_printf" },
	{ 0x656e4a6e, "snprintf" },
	{ 0x85dd963f, "seq_putc" },
	{ 0xfcdacab5, "tty_standard_install" },
	{ 0x2d3385d3, "system_wq" },
	{ 0xc5b6f236, "queue_work_on" },
	{ 0x6b10bee1, "_copy_to_user" },
	{ 0xb2fd5ceb, "__put_user_4" },
	{ 0x167e7f9d, "__get_user_1" },
	{ 0x8f9c199c, "__get_user_2" },
	{ 0xc6cbbc89, "capable" },
	{ 0x14e0e135, "pcpu_hot" },
	{ 0xaad8c7d6, "default_wake_function" },
	{ 0x4afb2238, "add_wait_queue" },
	{ 0x1000e51, "schedule" },
	{ 0x37110088, "remove_wait_queue" },
	{ 0x31e14eb7, "usb_deregister_dev" },
	{ 0x6763ce3f, "tty_port_tty_get" },
	{ 0xdfd56218, "tty_vhangup" },
	{ 0x8009863a, "tty_kref_put" },
	{ 0x9c817372, "tty_unregister_device" },
	{ 0xe2c17b5d, "__SCT__might_resched" },
	{ 0xfe487975, "init_wait_entry" },
	{ 0x8c26d495, "prepare_to_wait_event" },
	{ 0x92540fbf, "finish_wait" },
	{ 0x8ddd8aad, "schedule_timeout" },
	{ 0x69acdf38, "memcpy" },
	{ 0xe5bd83e6, "usb_autopm_get_interface_async" },
	{ 0xaeb28e47, "usb_anchor_urb" },
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0x34db050b, "_raw_spin_lock_irqsave" },
	{ 0xd35cce70, "_raw_spin_unlock_irqrestore" },
	{ 0x5b8239ca, "__x86_return_thunk" },
	{ 0xf7bdf60f, "usb_submit_urb" },
	{ 0x7671ec45, "_dev_err" },
	{ 0x87a21cb3, "__ubsan_handle_out_of_bounds" },
	{ 0x698f35fa, "usb_autopm_put_interface_async" },
	{ 0x63cc8f50, "usb_autopm_get_interface_no_resume" },
	{ 0x769ed3d5, "usb_autopm_put_interface" },
	{ 0xedd45d37, "usb_get_from_anchor" },
	{ 0x105d07b6, "usb_kill_urb" },
	{ 0x8427cc7b, "_raw_spin_lock_irq" },
	{ 0x4b750f53, "_raw_spin_unlock_irq" },
	{ 0xef427d21, "tty_port_put" },
	{ 0x4a18e87e, "usb_find_interface" },
	{ 0x769de1c6, "__tty_alloc_driver" },
	{ 0x67b27ec1, "tty_std_termios" },
	{ 0xb891dd44, "tty_register_driver" },
	{ 0x7d8ab275, "usb_register_driver" },
	{ 0x122c3a7e, "_printk" },
	{ 0x939664f0, "tty_unregister_driver" },
	{ 0xfe459a2f, "tty_driver_kref_put" },
	{ 0xeb233a45, "__kmalloc" },
	{ 0x88db9f48, "__check_object_size" },
	{ 0x13c49cc2, "_copy_from_user" },
	{ 0x37a0cba, "kfree" },
	{ 0x97e8bf8b, "usb_autopm_get_interface" },
	{ 0x6370edc3, "usb_control_msg" },
	{ 0x6ebe366f, "ktime_get_mono_fast_ns" },
	{ 0x50e5b3dc, "__tty_port_tty_hangup" },
	{ 0xe2964344, "__wake_up" },
	{ 0x2679d9cb, "__dynamic_dev_dbg" },
	{ 0x4dfa8d4b, "mutex_lock" },
	{ 0x7665a95b, "idr_remove" },
	{ 0x3213f038, "mutex_unlock" },
	{ 0xaaf606ea, "usb_put_intf" },
	{ 0x62406193, "tty_port_tty_wakeup" },
	{ 0x5df552c, "tty_port_hangup" },
	{ 0x8b989402, "tty_port_close" },
	{ 0x2f93810, "module_layout" },
};

MODULE_INFO(depends, "");

MODULE_ALIAS("usb:v1A86p55D2d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v1A86p55D3d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v1A86p55D5d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v1A86p55D6d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v1A86p55DAd*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v1A86p55DBd*dc*dsc*dp*ic*isc*ip*in00*");
MODULE_ALIAS("usb:v1A86p55DDd*dc*dsc*dp*ic*isc*ip*in00*");
MODULE_ALIAS("usb:v1A86p55DEd*dc*dsc*dp*ic*isc*ip*in00*");
MODULE_ALIAS("usb:v1A86p55DEd*dc*dsc*dp*ic*isc*ip*in02*");
MODULE_ALIAS("usb:v1A86p55E7d*dc*dsc*dp*ic*isc*ip*in00*");
MODULE_ALIAS("usb:v1A86p55D8d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v1A86p55D4d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v1A86p55D7d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v1A86p55DFd*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v1A86p55E9d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v1A86p55EAd*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v1A86p55E8d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v1A86p55EBd*dc*dsc*dp*ic*isc*ip*in00*");
MODULE_ALIAS("usb:v1A86p55ECd*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v1A86p55EFd*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v1A86p5610d*dc*dsc*dp*ic*isc*ip*in01*");

MODULE_INFO(srcversion, "83EBAF82FB405309B9A018A");
