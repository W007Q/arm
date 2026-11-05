#!/usr/bin/env bash
# 安装 WCH 官方 CH343 驱动，让设备生成 /dev/ttyCH343USB0。
# 本脚本不打开串口、不启动 ROS、不发送电机命令。
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "用法：./scripts/install_ch343_driver.sh（安装时请求 sudo 密码）"
    exit 0
fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
driver_version="2.2-b5515b91110f"
driver_sources="$project_root/artifacts/drivers/ch343/$driver_version"
kernel_release="$(uname -r)"
for source_file in ch343.c ch343.h Makefile dkms.conf; do
    [[ -f "$driver_sources/$source_file" ]] || { echo "缺少驱动文件：$driver_sources/$source_file" >&2; exit 1; }
done
[[ -d "/lib/modules/$kernel_release/build" ]] || { echo "缺少内核头文件：linux-headers-$kernel_release" >&2; exit 1; }
command -v dkms >/dev/null || { echo "缺少 DKMS，请先执行：sudo apt install dkms" >&2; exit 1; }
if (( EUID != 0 )); then
    exec sudo -- /bin/bash "$0" "$@"
fi

# 注册 DKMS：后续内核更新时自动重新编译，避免只装在旧内核下。
install -d "/usr/src/ch343-$driver_version"
for source_file in ch343.c ch343.h Makefile dkms.conf; do
    install -m 0644 "$driver_sources/$source_file" "/usr/src/ch343-$driver_version/$source_file"
done
if [[ ! -e "/var/lib/dkms/ch343/$driver_version/source" ]]; then
    dkms add -m ch343 -v "$driver_version"
fi
dkms install -m ch343 -v "$driver_version" -k "$kernel_release" --force

# 备份并停用旧规则中的 CH343 换绑段；其他型号保持原样。
python3 - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import shutil

rule = Path('/etc/udev/rules.d/99-ch34x.rules')
if rule.is_file():
    original = rule.read_text()
    lines = original.splitlines(keepends=True)
    output = []
    i = 0
    while i < len(lines):
        end = i + 1
        while end < len(lines) and lines[end - 1].rstrip().endswith('\\'):
            end += 1
        block = ''.join(lines[i:end])
        if (lines[i].lstrip().startswith('ACTION==')
                and 'ATTRS{idVendor}=="1a86"' in block
                and 'ATTRS{idProduct}=="55d3"' in block
                and '/drivers/cdc_acm/unbind' in block
                and '/drivers/usb_ch343/bind' in block):
            block = '# CH343 换绑由 98-rebotarm-ch343.rules 处理。\n' + ''.join(
                '# ' + line for line in lines[i:end])
        output.append(block)
        i = end
    updated = ''.join(output)
    if updated != original:
        backup = rule.with_name(rule.name + '.bak.' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        shutil.copy2(rule, backup)
        rule.write_text(updated)
        print('旧规则备份：', backup)
PY

# 厂商驱动存在时才换绑；只操作 CH343 的控制接口。
udev_rule="/etc/udev/rules.d/98-rebotarm-ch343.rules"
if [[ -f "$udev_rule" ]]; then
    cp -a -- "$udev_rule" "$udev_rule.bak.$(date -u +%Y%m%dT%H%M%S%NZ)"
fi
cat > "$udev_rule" <<'EOF'
# CH343 使用 WCH 官方驱动；允许 dialout 用户读写。
ACTION=="add", SUBSYSTEM=="usb", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTR{bInterfaceNumber}=="00", DRIVER=="cdc_acm", TEST=="/sys/bus/usb/drivers/usb_ch343/bind", RUN+="/bin/sh -c 'echo %k > /sys/bus/usb/drivers/cdc_acm/unbind'", RUN+="/bin/sh -c 'echo %k > /sys/bus/usb/drivers/usb_ch343/bind'"
SUBSYSTEM=="tty", KERNEL=="ttyCH343USB[0-9]*", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0660", GROUP="dialout"
EOF
install -d /etc/modules-load.d
echo ch343 > /etc/modules-load.d/rebotarm-ch343.conf
udevadm control --reload-rules
modprobe ch343

# 恢复当前已插着的 CH343，无需复位集线器或重新给电机通电。
python3 - <<'PY'
from pathlib import Path

for device in Path('/sys/bus/usb/devices').iterdir():
    if not (device / 'idVendor').exists():
        continue
    if ((device / 'idVendor').read_text().strip(),
            (device / 'idProduct').read_text().strip()) != ('1a86', '55d3'):
        continue
    for interface in device.glob(device.name + ':*'):
        if (interface / 'bInterfaceNumber').read_text().strip() != '00':
            continue
        driver = interface / 'driver'
        if driver.exists() and driver.resolve().name == 'cdc_acm':
            Path('/sys/bus/usb/drivers/cdc_acm/unbind').write_text(interface.name)
        if not driver.exists():
            Path('/sys/bus/usb/drivers/usb_ch343/bind').write_text(interface.name)
            print('已绑定 WCH 驱动：', interface.name)
PY
udevadm trigger --action=add --subsystem-match=tty --sysname-match='ttyCH343USB*'
udevadm settle --timeout=10
if [[ -c /dev/ttyCH343USB0 ]]; then
    ls -l /dev/ttyCH343USB0
    echo "CH343 已恢复，可使用 ./scripts/start_uart_bridge.sh /dev/ttyCH343USB0"
else
    echo "驱动已安装；请重新插拔 CH343 USB，再检查 /dev/ttyCH343USB*。" >&2
    exit 1
fi
