#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
#
# User data for one F2 instance (f2.6xlarge, FPGA Developer AMI, the build
# instance profile): load the bring-up AFI (AGFI), build the host tool
# (platforms/f2/host/) against AWS's SDK, run M8's measurements
# (scripts/f2/bringup_plan.py) and copy the results (JSON lines with
# provenance) and the log to S3 after every step. The instance powers off
# when done or at the deadline. Root and installs are allowed on the
# project's own instances (AGENTS.md, rule 2).

REF=main                       # branch of github.com/cattabiani/cabosse; set another at launch
AWS_FPGA_REF=f2                # branch of github.com/aws/aws-fpga
AGFI=                          # the bring-up AFI's global id (agfi-...)
AFI=                           # its id (afi-...), for the provenance
QUICK=                         # --quick: a short plan (smaller runs, fewer samples)
DEADLINE_MIN=60                # safety net: power off even if something hangs
S3=                            # s3://bucket/dcp/<run>: everything goes there (the role writes dcp/ only)

shutdown -h +$DEADLINE_MIN
export HOME=/root
out=/var/log/cabosse.log
exec > >(tee -a $out > /dev/console) 2>&1
log=/var/log/cabosse-setup.log
[ -n "$S3" ] && [ -n "$AGFI" ] || { echo "CABOSSE-RESULT fail S3 and AGFI must be set"; shutdown -h now; exit 1; }

save() { aws s3 cp --only-show-errors $out "$S3/cabosse.log"; }
finish() {
  save
  echo "CABOSSE-FINISHED"
  sleep 30
  shutdown -h now
  exit
}
{ while sleep 60; do save; done & } >>$log 2>&1

echo "CABOSSE-BEGIN setup"
date -u +%FT%TZ
cd /root || exit 1
git clone -q --depth 1 -b $REF https://github.com/cattabiani/cabosse.git >>$log 2>&1
commit=$(git -C cabosse rev-parse HEAD)
echo "cabosse $commit"
git clone -q --depth 1 -b $AWS_FPGA_REF https://github.com/aws/aws-fpga.git >>$log 2>&1
aws_fpga=$(git -C aws-fpga rev-parse HEAD)
echo "aws-fpga $aws_fpga"
source /etc/profile >>$log 2>&1
cd aws-fpga && { source sdk_setup.sh >>$log 2>&1 || { echo "sdk_setup failed"; tail -40 $log; finish; }; }
# <print> needs GCC 14; Ubuntu 24.04's default is 13.
{ apt-get update -q && apt-get install -y -q g++-14; } >>$log 2>&1 || { echo "g++-14 install failed"; tail -20 $log; finish; }
make -C /root/cabosse/platforms/f2/host CXX=g++-14 >>$log 2>&1 || { echo "host tool build failed"; tail -40 $log; finish; }
echo "CABOSSE-END setup"

echo "CABOSSE-BEGIN load"
fpga-clear-local-image -S 0 -H
fpga-load-local-image -S 0 -I $AGFI -H || { echo "CABOSSE-RESULT fail AFI load"; finish; }
fpga-describe-local-image -S 0 -R -H | tee /root/describe.txt
echo "CABOSSE-END load"

echo "CABOSSE-BEGIN plan"
token=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
meta() { curl -s -H "X-aws-ec2-metadata-token: $token" "http://169.254.169.254/latest/meta-data/$1"; }
export S3
PROVENANCE=$(python3 -c 'import json,sys; print(json.dumps(dict(zip(sys.argv[1::2], sys.argv[2::2]))))' \
  commit "$commit" aws_fpga "$aws_fpga" agfi "$AGFI" afi "$AFI" instance "$(meta instance-id)" \
  instance_type "$(meta instance-type)" az "$(meta placement/availability-zone)" \
  cpu "$(grep -m1 '^model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')")
export PROVENANCE
python3 /root/cabosse/scripts/f2/bringup_plan.py /root/cabosse/platforms/f2/host/bringup \
  /root/bringup.jsonl $QUICK
echo "plan exit $?"
aws s3 cp --only-show-errors /root/describe.txt "$S3/describe.txt"
grep -c '"error"' /root/bringup.jsonl | sed 's/^/steps with an error: /'
echo "CABOSSE-END plan"
echo "CABOSSE-RESULT done"
finish
