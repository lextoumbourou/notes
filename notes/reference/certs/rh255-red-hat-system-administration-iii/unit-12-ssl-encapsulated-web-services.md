---
title: Red Hat System Administration III - Unit 12 - SSL Encapsulated Web Services
date: 2013-08-10 00:00
type: course
category: reference
status: draft
tags:
  - Linux
---

* Need to install ```yum install mod_ssl``` or ```yum groupinstall web-server```
* When installed ```mod_ssl``` you can find the key config under ```/etc/httpd/conf.d/ssl.conf```
* When server first connects, it will complain that it doesn't have the public key
* Self-signed certificates have same subject and issuer
* openssl command examples to get public key cert info:

    ```openssl x509 -in example-ca.crt --text``` gets data on a certificate

## Customizing a Self-signed a Certificate

* Install ```crypto-utils```

    > yum install crypto-utils mod_ssl

* Use ```genkey``` to generate new public and private key

        > genkey --days 365 server2.example.com
        # Choose location
        # Choose key size
        /usr/bin/keyutil -c makecert -g 1024 -s "CN=server2.example.com, O=RedHat Inc\., L=Melbourne, ST=Victoria, C=AU" -v 12 -a -z /etc/pki/tls/.rand.3240 -o /etc/pki/tls/certs/server2.example.com.crt -k /etc/pki/tls/private/server2.example.com.key
        cmdstr: makecert

        cmd_CreateNewCert
        command:  makecert
        keysize = 1024 bits
        subject = CN=server2.example.com, O=RedHat Inc\., L=Melbourne, ST=Victoria, C=AU
        valid for 12 months
        random seed from /etc/pki/tls/.rand.3240
        output will be written to /etc/pki/tls/certs/server2.example.com.crt
        output key written to /etc/pki/tls/private/server2.example.com.key


        Generating key. This may take a few moments...

        Made a key
        Opened tmprequest for writing
        (null) Copying the cert pointer
        Created a certificate
        Wrote 882 bytes of encoded data to /etc/pki/tls/private/server2.example.com.key 
        Wrote the key to:
        /etc/pki/tls/private/server2.example.com.key

* With the paths returned, update ```/etc/httpd/conf.d/ssl.conf``` as follows:

        > vi /etc/httpd/conf.d/ssl.conf
        + SSLCertificateKeyFile /etc/pki/tls/private/server2.example.com.key
        + SSLCertificateFile /etc/pki/tls/certs/server2.example.com.crt

## Generating a Certificate Signing Requests

* Use ```--genreq``` option when running ```genkey```
* The cls will be stored in: ```/etc/pki/tls/tls/certs```, send that to Verisign or whatevs and they'll return you a key
