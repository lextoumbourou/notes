---
title: Red Hat System Administration III - Unit 3 - Bash Scripting and Tools
date: 2013-08-10 00:00
type: course
category: reference
status: draft
tags:
  - Linux
---

## Bash Overview

* Start scripts with `#!/bin/bash`
* Make it executable with `chmod`
* Put it in a directory in `$PATH`
    * `~/bin` - for user's private programs
    * `/usr/local/bin` - locally developed scripts by other uses on the system
    * `/usr/local/sbin` - locally developed scripts used by root
* Variable syntax:
    * `$my_var`
    * `${my_var}`
* Command substitution:
    * `today=$(date +%Y-%m-%d); echo $today`
* Quoting & escaping:
    * `echo "Today is $X"` - interpolate
    * `echo 'Today is $X'` - literal
* Repetition can be achieved with: `for var in commands; do command; done`

    `for i in {1..9}; do echo "This is $i"; done`

* Conditional branching:
    * utilise non-zero value returns for false
    * Syntax: `test EXPRESSION` or `test [ EXPRESSION ]`
    * Use `$?` to get last return code
    * Non-zero or zero-length string operators: `test -{n|z} STRING`
    * String comparison: `==`, `!=`
    * Numeric comparison: `-eq`, `-ne`
    * File status operations: `test -{e|f|d|r|w|x|s} FILE`
        * `-s` - check for empty files
    * Logical operators:
        * `-o` - OR
        * `-a` - AND
        * `!` - NOT
    * `man test` to get info on Bash operators
* Use `env` to get list of environment variables
* Positional params:
    * `script.sh TEST DEMO` - `$0 $1 $2`
    * `$@`- all positional params
    * `$#` - count of params
* Running commands on remote machine:
    * Save file locally:
       `ssh user@host 'command1; command2' > log.local`
    * Save file remotely:
        `ssh user@host 'command1; command2 > log.remote'`

## Text Processing Tools

### diff

* Used to compare two files
* Args:
    * `-c` - display surrounding lines for context
    * `-u` - unified output format
    * `-r` - perform a recursive comparison of files

### patch

* Changes files from patches created with `diff -u`:
    * `$ diff -u samsreport.sh samsreport-2.sh >> my_patch`
* Patch file using the `patch` command

```bash
$ patch samsreport.sh < my_patch
patching file samsreport.sh
```

* Use `-b` option to specify a backup before patching. Will create a file with an `.orig` suffix:
    * `> samsreport.sh.orig`

### grep

* Search for patterns of text
* Args:
    * `-i` perform case-insensitive search
    * `-n` precede return lines with line of numbers
    * `-r` perform a recursive search of lines, starting with the named directory
    * `-c` display a count of lines matching
    * `v` return lines that don't match
    * `-l` list names of files with one line containing the pattern

### cut

* Get first column delimitered by `:`:
    * ```> cat /etc/passwd | cut -d : -f 1```

### wc

* `-l` displays number of lines
* `-w` displays number of words
* `-c` displays number of bytes
* `-m` displays nubmer of characters

### sort

* `-n` - sort numerically instead of char
* `-k` - set the sort field
* `-t` - specify a field separator

### tr

Translate command

```bash
> echo "HELLO" | tr 'A-Z' 'a-z'
hello
```

### sed

* Edits a stream of textual data
    * Use ```g``` to make it global, use ```d``` to delete lines

```bash
> echo "dogs and cats" | sed '/ddogs/d'
dogs and cats
> echo "dogs and cats" | sed '/dogs/d'
```

### Regular Expression

* `^` - line begins
* `$` - line ends
* `.` - any single character
* `[z]` - a single character that is x, y or z
* `[^xyz]` - opposite

### Password Aging

Use `change` to change password information for a user

Defaults stored in `/etc/login.defs`:

* `-m` - min days
* `-M` - max days
* `-W` - warn days

Defaults stored in `/etc/default/useradd`:

* `-I` - inactive days
* `-E` - expiry
