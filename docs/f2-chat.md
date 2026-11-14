# Chat with the model on F2

How to run a manual chat with SmolLM2-135M-Instruct on the Cabosse core on
an AWS F2 instance (M11 step 2). It costs about $2.50 an hour (f2.6xlarge,
eu-central-1, estimate). The instance powers off by itself after 120
minutes at most.

## Before you start

- AWS CLI logged in: `aws sso login --profile cabosse` (see
  [aws-setup.md](aws-setup.md)).
- Your local `main` is the same as GitHub's (`git pull`). The instance
  clones `main`, and the script refuses to launch if the two differ.
- No other f2.6xlarge running (the script checks).

## 1. Launch (on your machine)

```
scripts/f2/launch_chat.sh chat1
```

`chat1` names the run. Its results go to
`s3://cabosse-builds-628837661672/dcp/<date>-chat1/`. The first time, the
script creates the SSH key `~/.ssh/cabosse.pem` and a security group that
lets in SSH from your current IP address only. It ends by printing the
ssh command.

Options, as environment variables before the command: `DEADLINE_MIN=60`
(power off sooner), `MHZ=190` (core clock; 200 by default, below build 9's
~213 MHz), `PCIS_READ=stream32:4` (the logits read plan, below).

## 2. Log in and wait for setup (on the instance)

Run the ssh command the script printed:

```
ssh -i ~/.ssh/cabosse.pem -o StrictHostKeyChecking=accept-new ubuntu@<IP>
```

If it is refused, wait a minute: the instance is still booting. Then
follow the setup:

```
tail -f /var/log/cabosse.log
```

Setup takes about 10 minutes: it builds the tools, installs Python,
downloads the model and loads the FPGA. Then it runs the PCIS read bench:
one JSON line for each read plan, with its `mb_per_s`. Wait for
`CABOSSE-READY`, then stop `tail` with Ctrl-C.

If you see `CABOSSE-RESULT fail`, setup failed: the log shows why. Power
off (step 5) and send the log.

## 3. Chat

```
cabosse-chat
```

Loading the model onto the card takes a little while. Then type a message
after `> ` and press Enter. The answer appears as it is generated,
followed by a line with its speed (`[N tokens, X tokens/s wall clock]`).
The model remembers the conversation until about 2,000 tokens, then
starts a new one (it says so).

`/quit` (or Ctrl-D) ends the chat. It is saved to S3 under `chat-manual/`
and the time of the power-off is printed. `cabosse-chat` again starts a
new chat (the model is loaded again).

## 4. Try a faster logits read (optional)

The bench lines in step 2 show which read plan is fastest (highest
`mb_per_s`, `"ok": true`). To use it for the next chat, for example
`stream32:4`:

```
echo 'export CABOSSE_PCIS_READ=stream32:4' | sudo tee -a /root/chat.env
cabosse-chat
```

Compare the tokens/s lines with the first chat's.

## 5. Power off

```
sudo shutdown -h now
```

This ends the instance and its cost. To keep it longer than planned,
`sudo shutdown -h +60` moves the power-off to 60 minutes from now.

If you lose the SSH session, the instance still powers off at its
deadline. To end it sooner from your machine:

```
aws ec2 describe-instances --profile cabosse --region eu-central-1 \
  --filters Name=instance-state-name,Values=running \
  --query 'Reservations[].Instances[].InstanceId' --output text
aws ec2 terminate-instances --profile cabosse --region eu-central-1 --instance-ids <id>
```
