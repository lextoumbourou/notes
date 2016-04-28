# Route53

## DNS 101

* IPv4
  * 32-bit field.
  * 4 billion addreses
* IPv6
  * Created to solve depletion issue.
  * 128 bits
  * 430 undecillion addresses

* Top-Level Domains
  * top-level - ``.com``
  * 2nd-level, top-level - ``.com.au``

* SOA Records
  * Stores the name of server that supplied data for the zone.
* NS Records
  * Name servers records used by top-level domain servers to direct traffic to DNS server with authoritative DNS records.
* A Record
  * Map name to IP.
* MX Record
  * For email.
* TTL
  * Length of time DNS record is cached on resolving server or user's own PC.
* CNAMES
  * Resolve one domain name to another.
* Alias
  * AWS created
  * Similar to CNAME but can be used for naked domain name.
  * Can map resource sets to ELBs, CloudFront distributions or S3 buckets. 

## Routing Policies

* Simple
  * Default routing policy when you create a record set.
  * Use when you have a single web server etc.
* Weighted
  * AB routing, 20% to one web server, 80% to other etc.
* Latency
  * Send to region with lowest latency for each user.
* Failover
  * Active / passive failover
* Geolocation
  * Send traffic depending on user's location.
