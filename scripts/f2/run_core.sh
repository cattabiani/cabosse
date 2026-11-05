#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
#
# User data for one F2 instance (f2.6xlarge, FPGA Developer AMI, the build
# instance profile): load the core's AFI (AGFI), build the host tools
# (platforms/f2/host/: core_driver, core_clock) against AWS's SDK, make a
# Python venv (uv: Python and requirements.txt as this project pins them),
# download SmolLM2 at the fixtures' revision, and run M9's measurements
# (scripts/f2/core_plan.py), copying the results (JSON lines with
# provenance) and the log to S3 as they come; then the chat recording
# (scripts/f2/chat.py) to $S3/chat/, and the CPU baselines
# (scripts/f2/cpu_baseline.sh) to $S3/baseline/. The instance powers off when
# done or at the deadline. Root and installs are allowed on the project's
# own instances (AGENTS.md, rule 2).

REF=main                       # branch of github.com/cattabiani/cabosse; set another at launch
AWS_FPGA_REF=f2                # branch of github.com/aws/aws-fpga
AGFI=                          # the core AFI's global id (agfi-...)
AFI=                           # its id (afi-...), for the provenance
MHZ=100                        # core clock: core1 closes timing at ~105 MHz, not 250
QUICK=                         # --quick: no long fixture
CHAT=1                         # the chat recording after the plan (empty: none)
BASELINE=1                     # the CPU baselines after the chat (empty: none)
DEADLINE_MIN=90                # safety net: power off even if something hangs
S3=                            # s3://bucket/dcp/<run>: everything goes there (the role writes dcp/ only)
UV_VERSION=0.12.7              # the installer of Python and the venv
PYTHON_VERSION=3.14.4
MODEL=HuggingFaceTB/SmolLM2-135M-Instruct
MODEL_REVISION=12fd25f77366fa6b3b4b768ec3050bf629380bac  # the fixtures' checkpoint

shutdown -h +$DEADLINE_MIN
export HOME=/root
out=/var/log/cabosse.log
exec > >(tee -a $out > /dev/console) 2>&1
log=/var/log/cabosse-setup.log
[ -n "$S3" ] && [ -n "$AGFI" ] || { echo "CABOSSE-RESULT fail S3 and AGFI must be set"; shutdown -h now; exit 1; }

save() {
  aws s3 cp --only-show-errors $out "$S3/cabosse.log"
  aws s3 cp --only-show-errors $log "$S3/setup.log"
}
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
cd aws-fpga && { source sdk_setup.sh >>$log 2>&1 || { echo "CABOSSE-RESULT fail sdk_setup failed"; tail -40 $log; finish; }; }
{ apt-get update -q && apt-get install -y -q g++-14; } >>$log 2>&1 || { echo "CABOSSE-RESULT fail g++-14 install failed"; tail -20 $log; finish; }
make -C /root/cabosse/platforms/f2/host CXX=g++-14 core_driver core_clock >>$log 2>&1 \
  || { echo "CABOSSE-RESULT fail host tools build failed"; tail -40 $log; finish; }
{ curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh | env UV_INSTALL_DIR=/root/.local/bin sh \
  && /root/.local/bin/uv venv -q -p $PYTHON_VERSION /root/venv \
  && /root/.local/bin/uv pip install -q -p /root/venv/bin/python --index-strategy unsafe-best-match -r /root/cabosse/requirements.txt; } >>$log 2>&1 \
  || { echo "CABOSSE-RESULT fail venv failed"; tail -40 $log; finish; }
export CABOSSE_WEIGHTS=/root/weights
/root/venv/bin/hf download $MODEL --revision $MODEL_REVISION --local-dir $CABOSSE_WEIGHTS/SmolLM2-135M-Instruct \
  >>$log 2>&1 || { echo "CABOSSE-RESULT fail model download failed"; tail -20 $log; finish; }
/root/venv/bin/python --version
echo "CABOSSE-END setup"

echo "CABOSSE-BEGIN load"
fpga-clear-local-image -S 0 -H
fpga-load-local-image -S 0 -I $AGFI -H || { echo "CABOSSE-RESULT fail AFI load"; finish; }
fpga-describe-local-image -S 0 -R -H | tee /root/describe.txt
aws s3 cp --only-show-errors /root/describe.txt "$S3/describe.txt"
echo "CABOSSE-END load"

echo "CABOSSE-BEGIN plan"
token=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
meta() { curl -s -H "X-aws-ec2-metadata-token: $token" "http://169.254.169.254/latest/meta-data/$1"; }
export S3
PROVENANCE=$(python3 -c 'import json,sys; print(json.dumps(dict(zip(sys.argv[1::2], sys.argv[2::2]))))' \
  commit "$commit" aws_fpga "$aws_fpga" agfi "$AGFI" afi "$AFI" model_revision "$MODEL_REVISION" \
  instance "$(meta instance-id)" instance_type "$(meta instance-type)" az "$(meta placement/availability-zone)" \
  cpu "$(grep -m1 '^model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')")
export PROVENANCE
/root/venv/bin/python /root/cabosse/scripts/f2/core_plan.py /root/core.jsonl --mhz $MHZ $QUICK
status=$?
echo "plan exit $status"
errors=$(grep -c '"error"' /root/core.jsonl)
echo "steps with an error: $errors"
echo "CABOSSE-END plan"
chat_status=0
if [ -n "$CHAT" ]; then
  echo "CABOSSE-BEGIN chat"
  S3=$S3/chat /root/venv/bin/python /root/cabosse/scripts/f2/chat.py /root/chat --mhz $MHZ
  chat_status=$?
  aws s3 cp --only-show-errors --recursive /root/chat "$S3/chat"
  echo "chat exit $chat_status"
  echo "CABOSSE-END chat"
fi
baseline_status=0
if [ -n "$BASELINE" ]; then
  echo "CABOSSE-BEGIN baseline"
  /root/cabosse/scripts/f2/cpu_baseline.sh /root/baseline
  baseline_status=$?
  aws s3 cp --only-show-errors --recursive /root/baseline "$S3/baseline"
  echo "baseline exit $baseline_status"
  echo "CABOSSE-END baseline"
fi
[ $status = 0 ] && [ "$errors" = 0 ] && [ $chat_status = 0 ] && [ $baseline_status = 0 ] && echo "CABOSSE-RESULT pass" || echo "CABOSSE-RESULT fail"
finish
