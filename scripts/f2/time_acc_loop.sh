#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
#
# User data for one CPU instance (FPGA Developer AMI, no FPGA needed): place
# and route platforms/f2/timing/acc_loop.sv on F2's part at 250 MHz, with
# the loop lengths of PIPE_REGS, all at once, and print the slack and the worst
# paths (M3, D-027). Results go to the serial console between CABOSSE
# markers, read back with `aws ec2 get-console-output --latest`. The instance
# powers off (and, launched with shutdown behaviour "terminate", terminates)
# when done or at the deadline.

REF=m3/acc-loop-timing         # branch of github.com/cattabiani/cabosse
AWS_FPGA_REF=f2                # branch of github.com/aws/aws-fpga, for the part name
FALLBACK_PART=xcvu47p-fsvh2892-2-e
PIPE_REGS="3 4 5 7"            # the FMA's registers; the loop adds one
RETIME=1                       # Vivado register retiming (acc_loop.tcl)
DEADLINE_MIN=55                # safety net: power off even if something hangs

shutdown -h +$DEADLINE_MIN
export HOME=/root  # cloud-init runs this without HOME; Vivado needs it
out=/var/log/cabosse.log
exec > >(tee -a $out > /dev/console) 2>&1
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
run nproc
run free -b
done_ host

section setup
cd /root || exit 1
git clone -q --depth 1 -b $REF https://github.com/cattabiani/cabosse.git >>$log 2>&1
git -C cabosse log -1 --format='cabosse %H %cI'
git clone -q --depth 1 -b $AWS_FPGA_REF https://github.com/aws/aws-fpga.git >>$log 2>&1
git -C aws-fpga log -1 --format='aws-fpga %H %cI'
source /etc/profile >>$log 2>&1
cd aws-fpga && { source hdk_setup.sh >>$log 2>&1 || { echo "hdk_setup failed"; tail -40 $log; }; }
cd /root
run which vivado
run vivado -version
part=$(grep -rhoE 'xcvu47p-[a-z0-9]+-[0-9]+-[a-z]' aws-fpga/hdk/common 2>/dev/null | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')
echo "part ${part:=$FALLBACK_PART}"
done_ setup

section vivado
date -u +%FT%TZ
pids=()
for n in $PIPE_REGS; do
  mkdir -p /root/run$n
  (
    cd /root/run$n || exit
    vivado -mode batch -nojournal -log vivado.log \
      -source /root/cabosse/platforms/f2/timing/acc_loop.tcl \
      -tclargs /root/cabosse "$part" "$n" "$RETIME" > vivado.out 2>&1
    status=$?
    echo "run$n exit $status $(date -u +%T)"
    [ $status -eq 0 ] || tail -30 vivado.out
  ) &
  pids+=($!)
done
wait "${pids[@]}"  # not a bare wait: it would also wait for the tee above
date -u +%FT%TZ
done_ vivado

for n in $PIPE_REGS; do
  section "loop $((n + 1))"
  cd /root/run$n || continue
  grep -h "CABOSSE-RESULT" vivado.log
  grep -E "^(ERROR|CRITICAL WARNING)" vivado.log | head -20
  sed -n '/Design Timing Summary/,/^$/p;/WNS(ns)/,+3p' timing_summary.rpt 2>/dev/null | head -12
  cat worst_paths.rpt 2>/dev/null | head -150
  grep -E "^\| (CLB LUTs|CLB Registers|CARRY8|DSPs) " utilization.rpt 2>/dev/null
  done_ "loop $((n + 1))"
done

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
