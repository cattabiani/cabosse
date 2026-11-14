#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
#
# Launch one f2.6xlarge for a manual chat on the core (M11 step 2): the
# user data is scripts/f2/run_core.sh with INTERACTIVE=1 (load the AFI, set
# the clock, run the PCIS read bench, install `cabosse-chat`), then print
# the SSH command. The instance powers off (and terminates) at the deadline,
# or earlier with `sudo shutdown -h now` on it.
#
# Usage: scripts/f2/launch_chat.sh NAME   (results go to $BUCKET/<date>-NAME)
# Works from any AWS account with F2 access (docs/f2-chat.md): the AFI is
# public (D-078). In the project's account, BUCKET and ROLE default to its
# results bucket and instance role; elsewhere both default to none, and
# the chats stay on the instance (/home/ubuntu/chats).
# Environment (defaults): PROFILE (cabosse, the AWS CLI profile), BUCKET
# (s3://..., the prefix results go under) and ROLE (an instance profile
# that can write it), AGFI and AFI (core build 9's), MHZ (200),
# DEADLINE_MIN (120), REF (main: the branch the instance clones; push it
# first), PCIS_READ (empty: the driver's default plan), BENCH (1);
# INTERACTIVE= (empty) runs the full measured run instead (M10's run 4;
# needs BUCKET and ROLE);
# DRY_RUN=1 prints the user data and calls nothing.
#
# On first use it makes the SSH key pair "cabosse" (private key in
# ~/.ssh/cabosse.pem) and the security group "cabosse-ssh"; each launch
# replaces that group's rules with SSH from this machine's public address
# only (Q-35). The subnet: the default one in the first availability zone
# that offers f2.6xlarge (SUBNET overrides).

set -euo pipefail

NAME=${1:?usage: launch_chat.sh NAME}
AGFI=${AGFI:-agfi-00e8542765d44a754}  # core build 9 (core9r2), 2026-11-14
AFI=${AFI:-afi-0c855f8fc0c08bb6b}
MHZ=${MHZ:-200}                        # below build 9's Fmax, about 213 MHz (D-075)
DEADLINE_MIN=${DEADLINE_MIN:-120}
REF=${REF:-main}
PCIS_READ=${PCIS_READ:-}
BENCH=${BENCH:-1}
INTERACTIVE=${INTERACTIVE-1}             # empty: run_core.sh's full plan, chat recording and baselines, then power off

PROFILE=${PROFILE:-cabosse}
REGION=eu-central-1                    # the AFI's region
TYPE=f2.6xlarge
AMI=ami-092ebd0b9baef5842              # FPGA Developer AMI (Ubuntu) 1.19.2 (AWS Marketplace, free), as the builds
PROJECT_ACCOUNT=628837661672           # the project's: its bucket and role are the defaults
KEY=cabosse
KEY_FILE=$HOME/.ssh/cabosse.pem
SG_NAME=cabosse-ssh
LOGIN=ubuntu                           # the AMI's user
REPO=$(cd "$(dirname "$0")/../.." && pwd)

aws() { command aws --profile "$PROFILE" --region "$REGION" "$@"; }

if [ -n "${DRY_RUN:-}" ]; then
  account=${ACCOUNT:-$PROJECT_ACCOUNT}
elif ! account=$(aws sts get-caller-identity --query Account --output text); then
  echo "the AWS CLI is not logged in: aws sso login --profile $PROFILE"
  exit 1
fi
if [ "$account" = "$PROJECT_ACCOUNT" ]; then
  BUCKET=${BUCKET-s3://cabosse-builds-628837661672/dcp}
  ROLE=${ROLE-cabosse-build-dcp}     # writes and reads $BUCKET/*
else
  BUCKET=${BUCKET:-}
  ROLE=${ROLE:-}
fi
S3=${BUCKET:+$BUCKET/$(date -u +%F)-$NAME}
[ -n "$S3" ] || [ -n "$INTERACTIVE" ] || { echo "a full run (INTERACTIVE=) needs BUCKET and ROLE"; exit 1; }
[ -z "$BUCKET" ] || [ -n "$ROLE" ] || { echo "BUCKET needs ROLE: an instance profile that can write it"; exit 1; }

# The user data: run_core.sh with this launch's settings, plain words only
# (they go into bash assignments and awk).
user_data=$(mktemp)
trap 'rm -f "$user_data" "$user_data.new"' EXIT
cp "$REPO/scripts/f2/run_core.sh" "$user_data"
for var in REF AGFI AFI MHZ DEADLINE_MIN S3 INTERACTIVE PCIS_READ BENCH; do
  [[ "${!var}" =~ ^[A-Za-z0-9._:/@+-]*$ ]] || { echo "$var=${!var}: letters, digits and ._:/@+- only"; exit 1; }
  grep -q "^$var=" "$user_data" || { echo "run_core.sh has no $var= line"; exit 1; }
  # The setting's line only (later lines may start with the name too).
  awk -v var="$var" -v val="${!var}" '!done && index($0, var "=") == 1 { sub(/^[^ ]*/, var "=" val); done = 1 } 1' \
    "$user_data" >"$user_data.new" && mv "$user_data.new" "$user_data"
done

if [ -n "${DRY_RUN:-}" ]; then  # the user data only, no AWS call
  cat "$user_data"
  exit 0
fi

# The instance clones REF: its scripts must be the ones here.
git -C "$REPO" fetch -q origin "$REF"
git -C "$REPO" diff --quiet FETCH_HEAD -- scripts/f2 platforms/f2/host sw model \
  || { echo "origin/$REF differs from this checkout in scripts/f2, platforms/f2/host, sw or model: push or set REF"; exit 1; }

# One F2 instance at a time (and the group's SSH rules belong to it).
running=$(aws ec2 describe-instances --filters "Name=instance-type,Values=$TYPE" \
  "Name=instance-state-name,Values=pending,running" --query 'Reservations[].Instances[].InstanceId' --output text)
[ -z "$running" ] || { echo "an $TYPE is running already: $running"; exit 1; }

# The key pair: made once; its private key lives only in KEY_FILE.
if ! keys=$(aws ec2 describe-key-pairs --filters "Name=key-name,Values=$KEY" --query 'KeyPairs[].KeyName' --output text); then
  echo "describe-key-pairs failed (aws sso login --profile $PROFILE?)"
  exit 1
fi
if [ -z "$keys" ]; then
  [ -e "$KEY_FILE" ] && { echo "$KEY_FILE exists but AWS has no key pair $KEY: move the file away first"; exit 1; }
  (umask 077 && aws ec2 create-key-pair --key-name "$KEY" --key-type ed25519 \
    --query KeyMaterial --output text >"$KEY_FILE.new") && mv "$KEY_FILE.new" "$KEY_FILE"
  echo "made key pair $KEY: $KEY_FILE"
elif [ ! -e "$KEY_FILE" ]; then
  echo "AWS has key pair $KEY but $KEY_FILE is missing: delete the pair (aws ec2 delete-key-pair --key-name $KEY) and rerun"
  exit 1
fi

# The subnet: a default one where F2 runs (default VPCs give public addresses).
if [ -z "${SUBNET:-}" ]; then
  azs=$(aws ec2 describe-instance-type-offerings --location-type availability-zone \
    --filters "Name=instance-type,Values=$TYPE" --query 'InstanceTypeOfferings[].Location' --output text)
  SUBNET=$(aws ec2 describe-subnets --filters Name=default-for-az,Values=true "Name=availability-zone,Values=${azs//$'\t'/,}" \
    --query 'sort_by(Subnets, &AvailabilityZone)[0].SubnetId' --output text)
  [ "$SUBNET" != None ] || { echo "no default subnet in $azs (aws ec2 create-default-vpc, or set SUBNET)"; exit 1; }
fi

# The security group: SSH from this machine's address only (no instance
# uses it now: checked above).
vpc=$(aws ec2 describe-subnets --subnet-ids "$SUBNET" --query 'Subnets[0].VpcId' --output text)
sg=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$vpc" \
  --query 'SecurityGroups[0].GroupId' --output text)
if [ "$sg" = None ]; then
  sg=$(aws ec2 create-security-group --group-name "$SG_NAME" --vpc-id "$vpc" \
    --description "Cabosse: SSH from the owner address only" --query GroupId --output text)
  echo "made security group $SG_NAME: $sg"
fi
ip=$(curl -fsS https://checkip.amazonaws.com)
[[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "no public IPv4 address found ($ip)"; exit 1; }
old=$(aws ec2 describe-security-group-rules --filters "Name=group-id,Values=$sg" \
  --query 'SecurityGroupRules[?!IsEgress].SecurityGroupRuleId' --output text)
[ -n "$old" ] && aws ec2 revoke-security-group-ingress --group-id "$sg" --security-group-rule-ids $old >/dev/null
aws ec2 authorize-security-group-ingress --group-id "$sg" --protocol tcp --port 22 --cidr "$ip/32" >/dev/null
echo "SSH allowed from $ip/32"

id=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --subnet-id "$SUBNET" \
  --security-group-ids "$sg" --key-name "$KEY" ${ROLE:+--iam-instance-profile "Name=$ROLE"} \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=120,VolumeType=gp3}' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=cabosse-$NAME}]" \
  --user-data "file://$user_data" --query 'Instances[0].InstanceId' --output text)
echo "launched $id ($TYPE, $REF, $AGFI at $MHZ MHz), powers off within $DEADLINE_MIN min; results: ${S3:-on the instance, /home/ubuntu/chats}"
aws ec2 wait instance-running --instance-ids "$id"
addr=$(aws ec2 describe-instances --instance-ids "$id" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
cat <<EOF
Setup takes about 3-10 minutes.
  ssh -i $KEY_FILE -o StrictHostKeyChecking=accept-new $LOGIN@$addr
  tail -f /var/log/cabosse.log    # wait for CABOSSE-READY
  cabosse-chat                    # /quit ends a chat; saved in /home/ubuntu/chats (and S3, with BUCKET)
  sudo shutdown -h +MIN           # move the power-off; 'now' ends (terminates) the instance
EOF
