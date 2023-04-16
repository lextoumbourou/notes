---
title: Red Hat System Administration III - Unit 5 - Package Management
date: 2013-08-10 00:00
type: course
category: reference
status: draft
tags:
  - Linux
---

## RPM

* Metadata stored in RPM that describes changes made in new version

To get information about an RPM use ```rpm -q```

```bash
rpm -q [select_options] [query_options]
```

1. ```-d``` or ```--docfiles```
2. ```-c``` or ```-configfiles```
3. ```-l``` or ```-list```
4. ```--scripts```

### Examples

```bash
> rpm -ql yum-plugin-verify
/etc/yum/pluginconf.d/verify.conf
/usr/lib/yum-plugins/verify.py
/usr/lib/yum-plugins/verify.pyc
/usr/lib/yum-plugins/verify.pyo
/usr/share/doc/yum-plugin-verify-1.1.30
/usr/share/doc/yum-plugin-verify-1.1.30/COPYING
/usr/share/man/man1/yum-verify.1.gz
```

Gets information on installed packages files.

## Yum

Yum plugins can be installed to extend functionality with ```yum-plugin-verify```

* ```yum verify <plugin_name>``` (from ```yum-plugin-verifiy```) verifies integrity of installed plugin

```bash
> rm /bin/mount
> yum verify util-linux-ng
==================== Installed Packages ====================
util-linux-ng.x86_64 : A collection of basic system utilities
    File: /bin/mount
        Problem:  file is missing
verify done
```

* ```yum-plugin-versionlock``` can lock a version to a certain version using config ```/etc/yum/pluginconf.d/version.conf```

```bash
> vi /etc/yum/plugincofn.d/versionlock.list
+ epoch:kernel-2.6.32-el6.x86_64
```

* ```yum whatprovides <filename>``` - used to check what package contains a file.
* ```yum reinstall <packagename>``` - reinstalls a package

## RPM Package Design

* 3 Basic Components:
    * metadata - data about the package: name, version, release, builder, date, deps
    * files - archive of files provided by package (inc file attributes)
    * scripts - execute when package is installed, updated and/or removed
* ```rpm2cpio <package_name> | cpio -id``` allows you to extract a rpm without installing them.

## RPM Package Specification

* Sequential order:

    1. Prep
    2. Build
    3. Install
    4. Clean

* Five steps for building and signing an RPM package:

    1. Tarball - get tar file containing source. By default, rpmbuild assumes the top-level of the archive is %{name}-%{version}. Place in ```~/rpmbuild/SOURCES``` directory
    2. Spec file - create spec file and populate required fields. Place in ```~/rpmbuild/SPECS```
    3. rpmbuild - use the rpmbuild command to build the packages ```rpmbuild -ba demo.spec```
    4. Sign - use GPG key to sign the RPM package. Use ```rpmbuild -ba --sign demo.spec```. If package is already built, usse ``rpm --design``` to add or change GPG sig
    5. Test - test package on dev system

* Creating a repo:
    * Use package ```yum install createrepo```
    * Create webdirectory:

```bash
> mkdir -p /var/www/html/example/Packages
> cp package_name.rpm /var/www/html/example/Packages
> createrepo -v /var/www/html/example
```

    * On clients, update ```/etc/yum.repos.d/example.repo```

```bash
> cat /etc/yum.repos.d/example.repo
[example]
name=example repo
baseurl=http://server2/example
enabled=1
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-demo
```
