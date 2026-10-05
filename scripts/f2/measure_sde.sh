#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
#
# User data for one f2.6xlarge (FPGA Developer AMI): load AWS's public cl_sde
# image and measure what it can show without a design of our own (the card,
# the PCIe link, the clocks, power, and host<->card streaming bandwidth).
# Results go to the serial console between CABOSSE markers, read back with
# `aws ec2 get-console-output --latest`. The instance powers off (and, launched
# with shutdown behaviour "terminate", terminates) when done or at the deadline.

AGFI=agfi-0925b211f5a81b071  # cl_sde, small shell 0x10212415 (hdk/cl/examples/cl_sde/README.md)
AWS_FPGA_REF=f2              # branch of github.com/aws/aws-fpga
DEADLINE_MIN=50              # safety net: power off even if something hangs
PACKET_SIZES="64 1024 4096 16384 65536"
REPEATS=3
MILLIONS_OF_PACKETS=1
HUGEPAGES=256

shutdown -h +$DEADLINE_MIN
out=/var/log/cabosse.log
exec > >(tee -a $out > /dev/console) 2>&1  # not to cloud-init too: the console would show it twice
log=/var/log/cabosse-build.log

section() { echo "CABOSSE-BEGIN $1"; }
done_() { echo "CABOSSE-END $1"; }
run() {  # run CMD...: print the command and its output, or note its failure
  echo "\$ $*"
  "$@" 2>&1 || echo "(exit $?)"
}

section host
date -u +%FT%TZ
run cat /etc/os-release
run uname -r
run lscpu
run free -b
done_ host

section setup
cd /root || exit 1
git clone -q --depth 1 -b $AWS_FPGA_REF https://github.com/aws/aws-fpga.git >>$log 2>&1
cd aws-fpga || { tail -20 $log; shutdown -h now; exit 1; }
git log -1 --format='aws-fpga %H %cI'
export HDK_DIR=$PWD/hdk
source sdk_setup.sh >>$log 2>&1 || { echo "sdk_setup failed"; tail -40 $log; }
done_ setup

section card-before-load
run fpga-describe-local-image -S 0 -H
run fpga-describe-local-image -S 0 -M
done_ card-before-load

section load
run fpga-load-local-image -S 0 -I $AGFI -H
run fpga-describe-local-image -S 0 -R -H
done_ load

section card
run fpga-describe-local-image -S 0 -M
run fpga-describe-clkgen -S 0
for dev in $(lspci -D -d 1d0f::0580 | cut -d' ' -f1); do run lspci -vv -s "$dev"; done  # the FPGA functions only
done_ card

section sde-build
pf0=$(lspci -D -d 1d0f:f002 | head -1 | cut -d' ' -f1)
echo "PF0 $pf0"
run setpci -s "$pf0" 4.w=6
run sysctl -w vm.nr_hugepages=$HUGEPAGES
cd "$HDK_DIR/cl/examples/cl_sde/software/runtime" || exit 1
make sde_c2h_perf_test sde_h2c_perf_test >>$log 2>&1 || { echo "build failed"; tail -40 $log; }
export LD_LIBRARY_PATH=$PWD/../src/sde_lib/lib/so:$LD_LIBRARY_PATH
done_ sde-build

for size in $PACKET_SIZES; do
  for direction in h2c c2h; do
    for repeat in $(seq $REPEATS); do
      section "sde $direction $size $repeat"
      run timeout 120 ./sde_${direction}_perf_test $MILLIONS_OF_PACKETS "$size" 0
      done_ "sde $direction $size $repeat"
    done
  done
done

section card-after
run fpga-describe-local-image -S 0 -M
done_ card-after

# The console keeps only its last 64 KiB: replay the log in pieces slow enough
# for a reader polling every 30 s to catch each one.
exec > /dev/console 2>&1
sleep 2  # let tee finish writing the log
split -b 16k -d $out /var/log/cabosse-part.
for part in /var/log/cabosse-part.*; do
  echo "CABOSSE-REPLAY ${part##*.}"
  cat "$part"
  echo
  sleep 40
done
echo "CABOSSE-FINISHED"
sleep 30  # let the console flush
shutdown -h now
