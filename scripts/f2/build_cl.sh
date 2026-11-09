#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
#
# User data for one CPU instance (FPGA Developer AMI): the full shell build
# of a CL, the bring-up design (platforms/f2/bringup/, PLAN.md M8) or the
# core (platforms/f2/core/, M9), as DESIGN says, with AWS's flow
# (aws_build_dcp_from_cl.py, Small Shell, recipe H0: clk_hbm_axi at 250 MHz,
# D-065), then platforms/f2/timing/report.tcl on the routed checkpoint. The
# log, the reports, the build's own log and the AFI tarball go to S3 as they
# appear; the AFI itself is made from the tarball afterwards (aws ec2
# create-fpga-image, by the operator). The run fails, with "CABOSSE-RESULT
# fail", if the route has negative slack (the shell flow's VIOLATED
# checkpoint; PLAN.md, M8: slack >= 0 at 250 MHz), or stops early (the
# watch, below). The instance powers off when done or at the deadline.

REF=main                       # branch of github.com/cattabiani/cabosse; set another at launch
AWS_FPGA_REF=f2                # branch of github.com/aws/aws-fpga
DESIGN=bringup                 # or core: platforms/f2/$DESIGN/, its top cabosse_$DESIGN.sv
CL=cabosse_$DESIGN
DEADLINE_MIN=$([ "$DESIGN" = core ] && echo 540 || echo 240)  # safety net: power off even if something hangs
S3=                            # s3://bucket/dcp/<build>: everything goes there (the role writes dcp/ only)
PLACE_WNS_MIN=-1.0             # ns: the build stops if placement ends below this (see "The watch")
PHYS_WNS_MIN=-0.25             # ns: ... or phys_opt_design after it

shutdown -h +$DEADLINE_MIN
export HOME=/root  # cloud-init runs this without HOME; Vivado needs it
export CL_DIR=/root/aws-fpga/hdk/cl/examples/$CL
out=/var/log/cabosse.log
exec > >(tee -a $out > /dev/console) 2>&1
log=/var/log/cabosse-build.log
[ -n "$S3" ] || { echo "CABOSSE-RESULT fail S3 not set: nothing would outlive the instance"; shutdown -h now; exit 1; }

save() {  # copy the logs so far to S3, over the last copies
  aws s3 cp --only-show-errors $out "$S3/cabosse.log"
  [ -d "$CL_DIR/build/scripts" ] && aws s3 cp --only-show-errors --recursive "$CL_DIR/build/scripts/" \
    "$S3/vivado/" --exclude "*" --include "*.log"
}
finish() {  # the last copies, then power off
  save
  echo "CABOSSE-FINISHED"
  sleep 30  # let the console flush
  shutdown -h now
  exit
}
{ while sleep 300; do save; done & } >>$log 2>&1

section() { echo "CABOSSE-BEGIN $1"; }
done_() { echo "CABOSSE-END $1"; }

section setup
date -u +%FT%TZ
nproc
free -b | head -2
cd /root || exit 1
git clone -q --depth 1 -b $REF https://github.com/cattabiani/cabosse.git >>$log 2>&1
commit=$(git -C cabosse rev-parse HEAD)
echo "cabosse $commit $(git -C cabosse log -1 --format=%cI)"
git clone -q --depth 1 -b $AWS_FPGA_REF https://github.com/aws/aws-fpga.git >>$log 2>&1
git -C aws-fpga log -1 --format='aws-fpga %H %cI'
source /etc/profile >>$log 2>&1
cd aws-fpga && { source hdk_setup.sh >>$log 2>&1 || { echo "hdk_setup failed"; tail -40 $log; }; }
vivado -version | head -1
done_ setup

section cl
# A CL directory from AWS's template (its build scripts and constraints),
# with our synthesis script in place of the template's.
cd /root/aws-fpga/hdk/cl/examples && python3 create_new_cl.py --new_cl_name $CL >>$log 2>&1
# hdk_setup.sh unsets CL_DIR when the directory does not exist yet (it is
# created just above): set it again for the rest and for AWS's build scripts.
export CL_DIR=/root/aws-fpga/hdk/cl/examples/$CL
design=$CL_DIR/design
[ -d "$design" ] || { echo "CABOSSE-RESULT fail create_new_cl.py"; tail -20 $log; finish; }
# The top in place of the template's, for AWS's scripts; synthesis reads our
# sources from the repository, in order (platforms/f2/synth_cl.tcl).
rm -f $design/*.sv $design/${CL}_defines.vh  # keep cl_id_defines.vh (AWS's PCI IDs, as ours)
cp /root/cabosse/platforms/f2/$DESIGN/$CL.sv /root/cabosse/platforms/f2/cl_shell.svh $design/
cp /root/cabosse/platforms/f2/synth_cl.tcl $CL_DIR/build/scripts/synth_$CL.tcl
# The design's floorplan, if it has one (read before placement, build_level_1_cl.tcl).
fp=/root/cabosse/platforms/f2/$DESIGN/small_shell_cl_pnr_user.xdc
[ -f $fp ] && cp $fp $CL_DIR/build/constraints/ && echo "floorplan $(wc -l < $fp) lines"
ls $design
done_ cl

section build
date -u +%FT%TZ
export CABOSSE_BUILD=${commit:0:8}
export CABOSSE_REPO=/root/cabosse
export CABOSSE_DESIGN=$DESIGN
cd $CL_DIR/build/scripts || { echo "CABOSSE-RESULT fail no build scripts"; finish; }
# Its own session (process group), so the watch can stop it with its Vivado.
setsid python3 aws_build_dcp_from_cl.py --cl $CL --mode small_shell --no-encrypt --aws_clk_gen \
  --clock_recipe_a A1 --clock_recipe_b B0 --clock_recipe_c C0 --clock_recipe_hbm H0 &
build_pid=$!
# The watch: stop the build early, saving its hours, if the floorplan does
# not apply (a critical warning on its file), or if the worst slack is
# below PLACE_WNS_MIN after placement or PHYS_WNS_MIN after phys_opt_design
# (each summary's first line; a later one comes inside route_design). Past
# builds: bring-up 4 -0.674 placed, +0.001 phys_opt, +0.069 routed (met);
# bring-up 3 -1.666, -1.195 (failed); core 2 -2.81, -2.706, -2.450. The
# checkpoint of the step that stopped it is kept for the reports.
abort= stop_dcp= linked=
first_wns() {  # the first number after $1 in the Vivado log, without a trailing dot
  local w
  w=$(grep -m1 -o "$1[-0-9.]*" $vlog | sed "s/.*=//; s/\.$//")
  echo "$w"
}
below() { awk -v w="$1" -v m="$2" 'BEGIN {exit !(w + 0 < m + 0)}'; }
placed= physd=
while kill -0 $build_pid 2>/dev/null; do
  sleep 60
  vlog=$(ls $CL_DIR/build/scripts/*.vivado.log 2>/dev/null | head -1)
  [ -n "$vlog" ] || continue
  if grep -q "^CRITICAL WARNING.*cl_pnr_user.xdc" $vlog; then
    abort="floorplan: $(grep -m1 '^CRITICAL WARNING.*cl_pnr_user.xdc' $vlog | cut -c1-300)"
    linked=1  # keep the linked checkpoint: the patterns can be checked on it
  fi
  if [ -z "$placed" ]; then
    wns=$(first_wns "Post Placement Timing Summary WNS=")
    if [ -n "$wns" ]; then
      placed=1
      echo "post-place WNS $wns ($(date -u +%FT%TZ))"
      below $wns $PLACE_WNS_MIN && abort="post-place WNS $wns < $PLACE_WNS_MIN" stop_dcp=post_place
    fi
  fi
  if [ -z "$physd" ]; then
    wns=$(first_wns "Post Physical Optimization Timing Summary | WNS=")
    if [ -n "$wns" ]; then
      physd=1
      echo "post-phys_opt WNS $wns ($(date -u +%FT%TZ))"
      below $wns $PHYS_WNS_MIN && abort="post-phys_opt WNS $wns < $PHYS_WNS_MIN" stop_dcp=post_phys_opt
    fi
  fi
  if [ -n "$abort" ]; then
    echo "stopping the build: $abort"
    if [ -n "$stop_dcp" ]; then  # its checkpoint first (up to 30 min)
      for i in $(seq 60); do
        grep -q "$stop_dcp.dcp' has been generated" $vlog && break
        sleep 30
      done
    fi
    kill -TERM -- -$build_pid
    sleep 30
    kill -KILL -- -$build_pid 2>/dev/null
    break
  fi
done
wait $build_pid
echo "build exit $?"
date -u +%FT%TZ
grep -hE "^(ERROR|CRITICAL WARNING)" $CL_DIR/build/scripts/*.vivado.log | sort | uniq -c | head -40
grep -ho "peak = [0-9.]*" $CL_DIR/build/scripts/*.vivado.log | sort -k3 -n | tail -1  # MB
done_ build

section result
cd $CL_DIR/build/checkpoints || { echo "CABOSSE-RESULT fail no checkpoints"; finish; }
ls -l
routed=$(ls *.post_route.dcp 2>/dev/null | head -1)
violated=$(ls *.post_route.VIOLATED.dcp 2>/dev/null | head -1)
stopped=$([ -n "$stop_dcp" ] && ls *.$stop_dcp.dcp 2>/dev/null | head -1)
linked=$([ -n "$linked" ] && ls *.post_link.dcp 2>/dev/null | head -1)
tarball=$(ls *.Developer_CL.tar 2>/dev/null | head -1)
# The build's own outputs first, before the reports' second Vivado session
# (its time could meet the deadline).
aws s3 cp --only-show-errors --recursive $CL_DIR/build/reports/ "$S3/shell-reports/"
[ -n "$tarball" ] && aws s3 cp --only-show-errors $CL_DIR/build/checkpoints/$tarball "$S3/$tarball" &&
  echo "tarball $S3/$tarball"
save
mkdir -p /root/reports && cd /root/reports
[ -n "$linked" ] && aws s3 cp --only-show-errors $CL_DIR/build/checkpoints/$linked "$S3/$linked" &&
  echo "checkpoint $S3/$linked"
if [ -n "$stopped" ]; then
  aws s3 cp --only-show-errors $CL_DIR/build/checkpoints/$stopped "$S3/$stopped" && echo "checkpoint $S3/$stopped"
fi
dcp=${routed:-${violated:-$stopped}}
if [ -n "$dcp" ]; then
  vivado -mode batch -nojournal -log report.log \
    -source /root/cabosse/platforms/f2/timing/report.tcl \
    -tclargs $CL_DIR/build/checkpoints/$dcp > report.out 2>&1
  # Where the blocks went (the floorplan's check): cells per SLR and more.
  [ "$DESIGN" = core ] && vivado -mode batch -nojournal -log analyze.log \
    -source /root/cabosse/platforms/f2/timing/analyze.tcl \
    -tclargs $CL_DIR/build/checkpoints/$dcp > analyze.out 2>&1 && cat cells_per_slr.rpt
  sed -n '/Design Timing Summary/,/^$/p;/WNS(ns)/,+3p' timing_summary.rpt | head -12
  head -60 worst_paths.rpt
  grep -E "^\| (CLB LUTs|CLB Registers|Block RAM|DSPs|  LUT as) " utilization.rpt
  sort -n families.rpt 2>/dev/null | head -30 | cut -c1-200
fi
if [ -n "$abort" ]; then
  echo "CABOSSE-RESULT fail stopped early: $abort"
elif [ -n "$routed" ] && [ -n "$tarball" ]; then
  echo "CABOSSE-RESULT pass $tarball"
elif [ -n "$violated" ]; then
  echo "CABOSSE-RESULT fail timing (negative slack after routing)"
else
  echo "CABOSSE-RESULT fail no routed checkpoint"
fi
aws s3 cp --only-show-errors --recursive /root/reports/ "$S3/reports/"
done_ result

finish
