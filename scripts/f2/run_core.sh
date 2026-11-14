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
# provenance) and the log to S3 as they come; then the PCIS read plans'
# rates (platforms/f2/host/pcis_bench.cpp) to $S3/pcis_bench.jsonl; then the chat recording
# (scripts/f2/chat.py) to $S3/chat/, and the CPU baselines
# (scripts/f2/cpu_baseline.sh) to $S3/baseline/. The instance powers off when
# done or at the deadline. Root and installs are allowed on the project's
# own instances (AGENTS.md, rule 2).
#
# INTERACTIVE=1 (M11's manual chat, scripts/f2/launch_chat.sh): no plan, chat
# recording or baselines; after the load it sets the clock (core_clock),
# runs the bench, installs `cabosse-chat` (chat.py --interactive, each chat
# to $S3/chat-manual/<time>/) and prints CABOSSE-READY. The instance then
# stays up for an SSH login until the deadline (`sudo shutdown -h +MIN`
# moves it).

REF=main                       # branch of github.com/cattabiani/cabosse; set another at launch
AWS_FPGA_REF=f2                # branch of github.com/aws/aws-fpga
AGFI=                          # the core AFI's global id (agfi-...)
AFI=                           # its id (afi-...), for the provenance
MHZ=100                        # core clock: core1 closes timing at ~105 MHz, not 250
QUICK=                         # --quick: no long fixture
BENCH=1                        # the PCIS read bench after the plan (empty: none)
PCIS_READ=                     # the driver's read plan (line_reader.h, e.g. stream32:4; empty: its default)
INTERACTIVE=                   # 1: set up for a manual chat over SSH, stay up until the deadline
CHAT=1                         # the chat recording after the plan (empty: none)
BASELINE=1                     # the CPU baselines after the chat (empty: none)
DEADLINE_MIN=90                # safety net: power off even if something hangs
S3=                            # s3://bucket/dcp/<run>: everything goes there (the role writes dcp/ only); INTERACTIVE may leave it empty
UV_VERSION=0.12.7              # the installer of Python and the venv
PYTHON_VERSION=3.14.4
# SmolLM2's logits (49,152 words) are 196,608 bytes; 9 reads a plan.
BENCH_PLANS="memcpy:1 load32:1 stream32:1 memcpy:4 load32:2 load32:4 load32:8 stream32:2 stream32:4 stream32:8 stream32:16"
MODEL=HuggingFaceTB/SmolLM2-135M-Instruct
MODEL_REVISION=12fd25f77366fa6b3b4b768ec3050bf629380bac  # the fixtures' checkpoint

shutdown -h +$DEADLINE_MIN
export HOME=/root
out=/var/log/cabosse.log
exec > >(tee -a $out > /dev/console) 2>&1
log=/var/log/cabosse-setup.log
[ -n "$AGFI" ] && { [ -n "$S3" ] || [ -n "$INTERACTIVE" ]; } \
  || { echo "CABOSSE-RESULT fail AGFI and S3 must be set (S3 may be empty with INTERACTIVE)"; shutdown -h now; exit 1; }

# put SRC PATH [aws s3 cp options]: SRC to $S3/PATH; nothing without S3.
put() { [ -z "$S3" ] || aws s3 cp --only-show-errors "${@:3}" "$1" "$S3/$2"; }
save() {
  put $out cabosse.log
  put $log setup.log
}
finish() {
  save
  echo "CABOSSE-FINISHED"
  sleep 30
  shutdown -h now
  exit
}
{ while sleep 60; do save; done & } >>$log 2>&1

# Progress: "[k/N] what (m:ss)" before each stage, N from the settings.
t0=$SECONDS
on() { [ -n "$1" ] && echo 1 || echo 0; }
total=$((5 + 1 + $(on "$BENCH")))  # setup and load; the clock or the plan; the bench
[ -z "$INTERACTIVE" ] && total=$((total + $(on "$CHAT") + $(on "$BASELINE")))
k=0
step() { k=$((k + 1)); printf '[%d/%d] %s (%d:%02d)\n' $k $total "$1" $(((SECONDS - t0) / 60)) $(((SECONDS - t0) % 60)); }

echo "CABOSSE-BEGIN setup"
step "clone the sources"
date -u +%FT%TZ
cd /root || exit 1
git clone -q --depth 1 -b $REF https://github.com/cattabiani/cabosse.git >>$log 2>&1
commit=$(git -C cabosse rev-parse HEAD)
echo "cabosse $commit"
git clone -q --depth 1 -b $AWS_FPGA_REF https://github.com/aws/aws-fpga.git >>$log 2>&1
aws_fpga=$(git -C aws-fpga rev-parse HEAD)
echo "aws-fpga $aws_fpga"
source /etc/profile >>$log 2>&1
step "AWS FPGA SDK and host tools"
cd aws-fpga && { source sdk_setup.sh >>$log 2>&1 || { echo "CABOSSE-RESULT fail sdk_setup failed"; tail -40 $log; finish; }; }
{ apt-get update -q && apt-get install -y -q g++-14; } >>$log 2>&1 || { echo "CABOSSE-RESULT fail g++-14 install failed"; tail -20 $log; finish; }
make -C /root/cabosse/platforms/f2/host CXX=g++-14 core_driver core_clock pcis_bench >>$log 2>&1 \
  || { echo "CABOSSE-RESULT fail host tools build failed"; tail -40 $log; finish; }
step "Python environment"
{ curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh | env UV_INSTALL_DIR=/root/.local/bin sh \
  && /root/.local/bin/uv venv -q -p $PYTHON_VERSION /root/venv \
  && /root/.local/bin/uv pip install -q -p /root/venv/bin/python --index-strategy unsafe-best-match -r /root/cabosse/requirements.txt; } >>$log 2>&1 \
  || { echo "CABOSSE-RESULT fail venv failed"; tail -40 $log; finish; }
export CABOSSE_WEIGHTS=/root/weights
step "download SmolLM2-135M-Instruct"
/root/venv/bin/hf download $MODEL --revision $MODEL_REVISION --local-dir $CABOSSE_WEIGHTS/SmolLM2-135M-Instruct \
  >>$log 2>&1 || { echo "CABOSSE-RESULT fail model download failed"; tail -20 $log; finish; }
/root/venv/bin/python --version
echo "CABOSSE-END setup"

echo "CABOSSE-BEGIN load"
step "load the FPGA image"
fpga-clear-local-image -S 0 -H
fpga-load-local-image -S 0 -I $AGFI -H || { echo "CABOSSE-RESULT fail AFI load"; finish; }
fpga-describe-local-image -S 0 -R -H | tee /root/describe.txt
put /root/describe.txt describe.txt
echo "CABOSSE-END load"

# PCIM (M11): the logits come back as the card's writes into a hugepage
# (core_driver.cpp): one 2 MiB page a driver, bus mastering on the CL's
# function (its PCI IDs, cl_shell.svh), and a record of the IOMMU (none
# expected: AWS's cl_sde wrote such pages by physical address, M8).
echo "CABOSSE-BEGIN pcim"
sysctl -w vm.nr_hugepages=4
grep -i huge /proc/meminfo
app_pf=$(lspci -D -d 1d0f:f010 | head -1 | cut -d' ' -f1)
echo "app PF $app_pf"
setpci -s "$app_pf" COMMAND=0x06:0x06 && lspci -vv -s "$app_pf" | grep -m1 Control:
ls /sys/class/iommu; dmesg | grep -i -m5 -e iommu -e amd-vi -e dmar
echo "CABOSSE-END pcim"

run_bench() {
  # The clock is set and no driver runs. It overwrites card memory at
  # address 0: a chat after it loads every weight again (Runtime.open).
  step "PCIS read bench"
  echo "CABOSSE-BEGIN bench"
  /root/cabosse/platforms/f2/host/pcis_bench 196608 9 $BENCH_PLANS | tee /root/pcis_bench.jsonl
  bench_status=${PIPESTATUS[0]}
  put /root/pcis_bench.jsonl pcis_bench.jsonl
  echo "bench exit $bench_status"
  echo "CABOSSE-END bench"
}

token=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
meta() { curl -s -H "X-aws-ec2-metadata-token: $token" "http://169.254.169.254/latest/meta-data/$1"; }
export S3
PROVENANCE=$(python3 -c 'import json,sys; print(json.dumps(dict(zip(sys.argv[1::2], sys.argv[2::2]))))' \
  commit "$commit" aws_fpga "$aws_fpga" agfi "$AGFI" afi "$AFI" model_revision "$MODEL_REVISION" \
  instance "$(meta instance-id)" instance_type "$(meta instance-type)" az "$(meta placement/availability-zone)" \
  cpu "$(grep -m1 '^model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')")
export PROVENANCE
[ -n "$PCIS_READ" ] && export CABOSSE_PCIS_READ=$PCIS_READ
bench_status=0

if [ -n "$INTERACTIVE" ]; then
  echo "CABOSSE-BEGIN interactive"
  step "set the clock to $MHZ MHz"
  /root/cabosse/platforms/f2/host/core_clock $MHZ | tee /root/clock.json
  [ "${PIPESTATUS[0]}" = 0 ] || { echo "CABOSSE-RESULT fail clock"; finish; }
  put /root/clock.json clock.json
  [ -n "$BENCH" ] && run_bench
  { printf 'export %s=%q\n' S3 "$S3" PROVENANCE "$PROVENANCE" CABOSSE_WEIGHTS "$CABOSSE_WEIGHTS" HF_HUB_OFFLINE 1
    [ -n "$PCIS_READ" ] && printf 'export CABOSSE_PCIS_READ=%q\n' "$PCIS_READ"; } > /root/chat.env
  cat > /root/chat.sh <<'CHAT_EOF'
#!/bin/bash
# One manual chat (run_core.sh INTERACTIVE=1), saved to S3 when it ends.
source /root/chat.env
dir=/home/ubuntu/chats/$(date -u +%Y%m%d-%H%M%S)  # the login user's: scp can fetch it
to=${S3:+$S3/chat-manual/${dir##*/}}  # empty without S3: the chat stays here
env S3="$to" /root/venv/bin/python /root/cabosse/scripts/f2/chat.py "$dir" --interactive "$@"
trap '' INT  # a second Ctrl-C must not stop the upload
chown -R ubuntu:ubuntu /home/ubuntu/chats
echo "saved in $dir"
[ -z "$to" ] || { aws s3 cp --only-show-errors --recursive "$dir" "$to" && echo "and in $to"; }
usec=$(sed -n 's/^USEC=//p' /run/systemd/shutdown/scheduled 2>/dev/null)
[ -n "$usec" ] && echo "power off at $(date -u -d @$((usec / 1000000)) +%H:%M) UTC; sudo shutdown -h +MIN moves it"
CHAT_EOF
  chmod 700 /root/chat.sh
  printf '#!/bin/bash\nexec sudo /root/chat.sh "$@"\n' > /usr/local/bin/cabosse-chat
  chmod 755 /usr/local/bin/cabosse-chat
  echo "bench exit $bench_status"
  echo "CABOSSE-READY: ssh in and run cabosse-chat"
  echo "CABOSSE-END interactive"
  save
  exit 0  # the instance stays up until the deadline
fi

echo "CABOSSE-BEGIN plan"
step "measurements (core_plan.py)"
/root/venv/bin/python /root/cabosse/scripts/f2/core_plan.py /root/core.jsonl --mhz $MHZ $QUICK
status=$?
echo "plan exit $status"
errors=$(grep -c '"error"' /root/core.jsonl)
echo "steps with an error: $errors"
echo "CABOSSE-END plan"
[ -n "$BENCH" ] && run_bench
chat_status=0
if [ -n "$CHAT" ]; then
  echo "CABOSSE-BEGIN chat"
  step "chat recording"
  S3=$S3/chat /root/venv/bin/python /root/cabosse/scripts/f2/chat.py /root/chat --mhz $MHZ
  chat_status=$?
  put /root/chat chat --recursive
  echo "chat exit $chat_status"
  echo "CABOSSE-END chat"
fi
baseline_status=0
if [ -n "$BASELINE" ]; then
  echo "CABOSSE-BEGIN baseline"
  step "CPU baselines"
  /root/cabosse/scripts/f2/cpu_baseline.sh /root/baseline
  baseline_status=$?
  put /root/baseline baseline --recursive
  echo "baseline exit $baseline_status"
  echo "CABOSSE-END baseline"
fi
[ $status = 0 ] && [ "$errors" = 0 ] && [ $bench_status = 0 ] && [ $chat_status = 0 ] && [ $baseline_status = 0 ] && echo "CABOSSE-RESULT pass" || echo "CABOSSE-RESULT fail"
finish
