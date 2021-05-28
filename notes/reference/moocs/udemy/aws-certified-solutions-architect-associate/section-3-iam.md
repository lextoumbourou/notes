# Identity Acces Management (IAM)

## Overview

* Access you to manage users and access.
* Shared Access to your AWS account.
* Granular permissions.
* Identity Federation (Active Directory, Facebook, LinkedIn auth).
* Mulifactor Auth.
* Temporary access for users/devices and services.
* Set your own password rotation policy.
* Integrates with many diff AWS services.
* Supports PCI DSS Compliance.

## Critical Terms

* Users - End users (people).
* Groups - A collection of users under one set of permissions.
* Roles - roles can be assigned to AWS resources.
* Policies - a document that defines one (or more permissions).

## IAM Lab notes

* IAM sign-in link.
  * Send staff members IAM sign-in link.
  * Can customize it by clicking "Customize".
    * Must choose unique name.

* Activate MFA on your root account.
  * Root account == account attached to email used to sign up.
  * Has complete Admin access. 

* Create indiviual IAM users.
  * By default, users have no permissions; need to grant permissions using **Policies**.
  * New Users get Access Key ID & Secret Access Keys for use via API and CMD.
  * Need to assign a password to login via the console.

* Policies
  * Create Groups then attach Policies for group.
  * For example, can attach ``AdministratorAccess`` to the ``Admins`` group.
  * JSON-object - key value pair.

* Role
  * Allow resources in AWS access other resources in AWS.
  * Can attach policies to roles.

* Policy Document example:

  ```
  {"Version": "2012-10-17",
  "Statement":
  [
    {"Effect": "Allow",
     "Action": "*",
     "Resource": "*"}
  ]}
  ```

* IAM is universal. Doesn't apply to regions so far.
