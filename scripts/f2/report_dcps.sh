#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
#
# User data for one CPU instance (FPGA Developer AMI): the reports of
# platforms/f2/timing/report.tcl on routed checkpoints saved by
# time_loops.sh, without a rebuild, all at once. The checkpoints come as
# presigned URLs (the build role may only write to S3); the reports and the
# log go to S3 (under dcp/*, where the role may write) and the key lines to
# the serial console. The instance powers off when done or at the deadline.

REF=main                       # branch of github.com/cattabiani/cabosse
AWS_FPGA_REF=f2                # branch of github.com/aws/aws-fpga, for Vivado's setup
DCPS=                          # name=URL ...: each checkpoint's (or AFI tarball's) presigned URL
TCL=platforms/f2/timing/report.tcl  # the reports to run, from the repository root
S3=                            # s3://bucket/dcp/<build>: reports go to $S3/<name>/
DEADLINE_MIN=90                # safety net: power off even if something hangs

shutdown -h +$DEADLINE_MIN
export HOME=/root  # cloud-init runs this without HOME; Vivado needs it
out=/var/log/cabosse.log
exec > >(tee -a $out > /dev/console) 2>&1
log=/var/log/cabosse-build.log

save_log() {  # copy the log so far to S3, over the last copy
  aws s3 cp --only-show-errors $out "$S3/report-dcps.log"
}
{ while sleep 300; do save_log; done & } >>$log 2>&1

echo "CABOSSE-BEGIN setup"
date -u +%FT%TZ
cd /root || exit 1
git clone -q --depth 1 -b $REF https://github.com/cattabiani/cabosse.git >>$log 2>&1
git -C cabosse log -1 --format='cabosse %H %cI'
git clone -q --depth 1 -b $AWS_FPGA_REF https://github.com/aws/aws-fpga.git >>$log 2>&1
source /etc/profile >>$log 2>&1
cd aws-fpga && { source hdk_setup.sh >>$log 2>&1 || { echo "hdk_setup failed"; tail -40 $log; }; }
cd /root
vivado -version | head -1
echo "CABOSSE-END setup"

pids=()
for entry in $DCPS; do
  name=${entry%%=*}
  mkdir -p /root/$name
  (
    cd /root/$name || exit
    curl -sSf -o routed.dcp "${entry#*=}" || { echo "$name: download failed"; exit 1; }
    if tar tf routed.dcp >/dev/null 2>&1; then  # an AFI tarball: its routed checkpoint
      tar xf routed.dcp --wildcards --strip-components 2 '*_routed.dcp' && mv -f ./*_routed.dcp routed.dcp
    fi
    vivado -mode batch -nojournal -log vivado.log \
      -source /root/cabosse/$TCL -tclargs routed.dcp > vivado.out 2>&1
    echo "$name exit $? $(date -u +%T)"
    aws s3 cp --only-show-errors --recursive . "$S3/$name/" --exclude "*" \
      --include "*.rpt" --include "vivado.*" && echo "$name saved to $S3/$name/"
  ) &
  pids+=($!)
done
wait "${pids[@]}"  # not a bare wait: it would also wait for the tee and the loop

for entry in $DCPS; do
  name=${entry%%=*}
  echo "CABOSSE-BEGIN $name"
  grep -E "^(ERROR|CRITICAL WARNING)" /root/$name/vivado.log | head -10
  grep "^CABOSSE-ANALYZE" /root/$name/vivado.log
  sed -n '/WNS(ns)/,+3p' /root/$name/timing_summary.rpt 2>/dev/null | head -4
  cat /root/$name/slack_histogram.rpt 2>/dev/null
  sort -n /root/$name/families.rpt 2>/dev/null | head -25 | cut -c1-200
  echo "CABOSSE-END $name"
done
echo "CABOSSE-FINISHED"
sleep 2  # let tee finish writing the log
save_log
sleep 30  # let the console flush
shutdown -h now
