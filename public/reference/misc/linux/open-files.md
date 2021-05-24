# Open files

To check limits of a PID:

```
cat /prod/<pid>/limits
```

To enforce limits for all users:

```
vi /etc/security/limits.conf
* soft nofile 65535
* hard nofile 65535
root soft nofile 65535
root hard nofile 65535
```

*Note: wildcard does not apply to root: it must be set manually.*
