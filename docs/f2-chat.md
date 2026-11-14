# Run Cabosse on AWS F2

How to chat with SmolLM2-135M-Instruct running on the Cabosse core, on an
AWS F2 FPGA instance in your own AWS account. You need no FPGA tools and
no build: the FPGA image (AFI) is public (D-078). One script launches the
instance; you log in with SSH and type.

Cost: an f2.6xlarge costs about $2.50 an hour (eu-central-1, on demand,
estimate; check AWS's price list). Setup takes about 5-10 minutes, and the
instance powers off by itself after 120 minutes at most. A 20-minute try is
about $1.

## 0. What you need

- **An AWS account with F2 access.** New accounts may run no F2 instance
  until AWS raises the quota "Running On-Demand F instances" to 24 vCPUs in
  eu-central-1 (Frankfurt). Follow [aws-setup.md](aws-setup.md), steps 1-6
  (account, root user, cost alarm, login, AWS CLI with a profile named
  `cabosse`, quota). The quota request can take hours or days.
- **The FPGA Developer AMI**, the instance's system image: in the AWS
  console, open AWS Marketplace, search "FPGA Developer AMI (Ubuntu)",
  choose **Continue to Subscribe** and accept the terms. It is free; you
  pay for the instance only. Without it the launch fails with
  `OptInRequired` and a link to the same page.
- **A Linux or macOS machine** with `bash`, `git`, `curl`, `ssh` and the AWS
  CLI v2. Nothing else: no Python, no FPGA tools.
- **Region eu-central-1.** The AFI exists there only. The script uses it
  whatever your CLI's default region is.

## 1. Get the project

```
git clone https://github.com/cattabiani/cabosse.git
cd cabosse
```

## 2. Launch the instance (on your machine)

```
aws sso login --profile cabosse
scripts/f2/launch_chat.sh chat1
```

`chat1` names the run. The first time, the script creates an SSH key,
`~/.ssh/cabosse.pem`, and a security group `cabosse-ssh` that lets in SSH
from your current IP address only. It refuses to start a second f2.6xlarge
while one is running. It ends by printing the ssh command.

Options, as environment variables before the command: `DEADLINE_MIN=60`
(power off sooner), `PROFILE=myprofile` (another AWS CLI profile),
`MHZ=190` (the core clock: 200 by default, below the design's ~213 MHz).

## 3. Log in and wait for setup (on the instance)

Run the ssh command the script printed, for example:

```
ssh -i ~/.ssh/cabosse.pem -o StrictHostKeyChecking=accept-new ubuntu@<IP>
```

If it is refused, wait a minute: the instance is still booting. Then follow
the setup:

```
tail -f /var/log/cabosse.log
```

Lines like `[4/7] download SmolLM2-135M-Instruct (2:10)` show the progress:
building the tools, installing Python, downloading the model, loading the
FPGA, setting its clock, and a short benchmark of the host's reads from the
card (one JSON line per method, with its MB/s). Wait for `CABOSSE-READY`,
then stop `tail` with Ctrl-C. If you see `CABOSSE-RESULT fail`, setup
failed and the log shows why: power off (step 5).

## 4. Chat

```
cabosse-chat
```

Loading the model onto the card takes a few seconds. Then type a message
after `> ` and press Enter. The answer appears as the card generates it,
followed by its speed (`[N tokens, X tokens/s wall clock]`; about 280
tokens/s). The model remembers the conversation up to about 2,000 tokens,
then starts a new one (it says so). Answers stop at 128 tokens.

SmolLM2-135M is a very small model: it is fast but often wrong, and it
tends to keep talking about the previous topic. Self-contained questions
work best.

`/quit` (or Ctrl-D) ends the chat. It is saved in `/home/ubuntu/chats/`
(the transcript, a replayable recording `chat.cast` and the timings). To
copy the chats to your machine before powering off:

```
scp -r -i ~/.ssh/cabosse.pem ubuntu@<IP>:chats .
```

`cabosse-chat --debug` also prints the tokens fed to the model each turn.

## 5. Power off

```
sudo shutdown -h now
```

This terminates the instance and stops its cost. To keep it longer than
planned, `sudo shutdown -h +60` moves the power-off to 60 minutes from now.

If you lose the SSH session, the instance still powers off at its
deadline. To end it sooner from your machine:

```
aws ec2 describe-instances --profile cabosse --region eu-central-1 \
  --filters Name=instance-state-name,Values=running \
  --query 'Reservations[].Instances[].InstanceId' --output text
aws ec2 terminate-instances --profile cabosse --region eu-central-1 --instance-ids <id>
```

The key pair and the security group stay in your account (they cost
nothing) for the next launch.

## For the project's account

In the project's own AWS account the script also saves every chat and the
setup log to its S3 bucket, and `INTERACTIVE= scripts/f2/launch_chat.sh
NAME` runs the full measured run instead (fixtures checked bit for bit,
the README's chat recording, CPU baselines), then powers off. Another
account can do the same with `BUCKET=s3://...` and `ROLE=` an instance
profile that can write that bucket.

`CABOSSE_PCIS_READ` chooses how the host reads the logits from the card
(`platforms/f2/host/line_reader.h`); the default, `stream32:8`, was the
fastest on the F2 host (reports/data/f2/2026-11-14-chat1.txt). To try
another for the next chat:

```
echo 'export CABOSSE_PCIS_READ=stream32:4' | sudo tee -a /root/chat.env
```
