# AWS account setup for Cabosse

A one-time setup, for someone new to AWS. It ends with a CLI login that the
project scripts can use, a cost alarm, and a request to AWS for F2 access.
Console menus move from time to time: if a name below has changed, search for
it in the console's search bar.

Cabosse runs in Frankfurt (`eu-central-1`), where F2 instances are available.

## 1. Create the account

1. Go to <https://aws.amazon.com> and choose **Create an AWS account**.
2. Give an email address only you control, and an account name (e.g.
   `cabosse`). The email becomes the **root user**: it can do anything,
   including close the account.
3. Enter a payment card and verify your phone number.
4. Choose the **Basic support** plan (free).
5. Choose the **Paid plan**, not the Free plan. The Free plan blocks
   expensive services (F instances are the kind it blocks) and closes the
   account after 6 months. Both start with the same sign-up credits, and
   "paid" only means you pay for what you use.

## 2. Protect the root user

The root user is for emergencies only. After this step you never use it day
to day.

1. Sign in as root, open the account menu (top right) → **Security
   credentials**.
2. Under **Multi-factor authentication (MFA)**, choose **Assign MFA device**.
   An authenticator app on your phone, or a passkey or security key.
3. Do not create access keys for the root user.

## 3. Set a cost alarm

AWS bills by the second while instances run. An alarm by email is the safety
net if something is left running.

1. In the console, open **Billing and Cost Management** → **Budgets** →
   **Create budget**.
2. Choose the **Monthly cost budget** template. Set the amount to the agreed
   cap and add your email. AWS sends alerts at 85% and 100% of it, and when
   the forecast exceeds it.
3. Optionally add a **Zero spend budget** too: an email as soon as anything
   costs money.
4. In **Cost Anomaly Detection**, create a monitor for AWS services, with
   email alerts. It is free.

A budget alarm does not stop anything. Stopping instances is still our job
(the project scripts do it and check it).

## 4. Create your everyday login (IAM Identity Center)

This gives you a separate user with MFA, and a CLI login that hands out
temporary credentials: no long-lived keys on disk.

1. Top right, set the region to **Europe (Frankfurt) eu-central-1**.
2. Open **IAM Identity Center** → **Enable**. Choose the option **with AWS
   Organizations** (an "organization instance"). It turns your account into
   the management account of an organization of one, at no cost. The other
   option, an "account instance", cannot give access to the AWS account
   itself. Note the **AWS access portal URL** shown on the dashboard
   (`https://….awsapps.com/start`).
3. **Users** → **Add user**: your name and email. You get an email to set a
   password and MFA. Do it.
4. **Permission sets** → **Create permission set** → **Predefined** →
   **AdministratorAccess**. (Admin is simplest for a one-person account. Once
   the scripts are settled, we can narrow it.)
5. **AWS accounts** → select your account → **Assign users or groups** → your
   user → the permission set.
6. Sign out of root. From now on, sign in through the access portal URL.

## 5. Install and log in with the AWS CLI

The project never installs anything system-wide; you install the CLI. Without
`sudo`, into your home directory:

```
curl -o /tmp/awscliv2.zip https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip
cd /tmp && unzip -q awscliv2.zip
./aws/install -i ~/.local/aws-cli -b ~/.local/bin
aws --version   # ~/.local/bin must be on PATH
```

(`unzip` must be installed; if it is missing, install it with your package
manager.) Then create a profile named `cabosse`:

```
aws configure sso --profile cabosse
```

Answer:
- **SSO session name:** `cabosse`
- **SSO start URL:** the access portal URL from step 4
- **SSO region:** `eu-central-1`
- **SSO registration scopes:** press Enter (the default)
- A browser opens: approve. Then choose the account and the
  `AdministratorAccess` role.
- **Default client Region:** `eu-central-1`; **output format:** `json`.

In the repo's local `.envrc` (never committed), add:

```
export AWS_PROFILE=cabosse
```

Each day, log in with `aws sso login`. Check it with
`aws sts get-caller-identity`, which prints your account and role.

Never paste keys, secrets, or tokens into a chat, an issue, or a file in the
repo.

## 6. Ask AWS for F2 access

New accounts can run no F instances: the quota "Running On-Demand F
instances" starts at 0 vCPUs. One f2.6xlarge needs 24.

Check it:

```
aws service-quotas get-service-quota --service-code ec2 --quota-code L-74FC7D96
```

To raise it: console, region Frankfurt, **Service Quotas** → **AWS
services** → **Amazon Elastic Compute Cloud (Amazon EC2)** → **Running
On-Demand F instances** → **Request increase at account level** → `24`.

A short, concrete reason helps. For example:

> Cabosse is an open-source hardware accelerator for LLM inference
> (Apache-2.0 and Solderpad licensed, public on GitHub). We need one
> f2.6xlarge in eu-central-1 for a few hours at a time: first to run AWS's
> own HDK example designs and measure HBM and DDR bandwidth, later to test
> our own design. Instances are started on demand and stopped after each
> session, under a monthly budget alarm.

AWS answers by email, from hours to days. A new account with little billing
history can be refused at first. If so, reply in the support case with the
same details and ask again later, or ask for a smaller first step.

## When this is done

Tell the agent: the profile name, that `aws sts get-caller-identity` works,
the quota value, and the budget cap. Nothing in the cloud is created until
the owner approves it, with that budget (AGENTS.md, rule 3).

## Sources

- [AWS Free Tier changes, July 2025](https://www.infoq.com/news/2025/07/aws-risk-free-account-credits/)
- [Creating an AWS account](https://docs.aws.amazon.com/accounts/latest/reference/manage-acct-creating.html)
- [Getting started with IAM Identity Center](https://docs.aws.amazon.com/singlesignon/latest/userguide/getting-started.html)
- [IAM Identity Center authentication with the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html)
- [Installing the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) and [install options](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-version.html)
- [EC2 instance type quotas](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html)
