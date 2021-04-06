Also known as [[Brewer's theorem]], after computer scientist Eric Brewer, is the theory that it's impossible for a distributed data store to provide 3 of the following capabilities simultaneously:

* Consistency: every read gets the most recent write
* Availability: Every read request receives a non-error response
* Partition tolerance: The system continues to operate despite some messages being dropped between nodes

Though often referred to as a "two out of three" tradeoff (you can have 2 out of 3 properties), it's actually more of a question about how the datastore deals with partitional intolerance: does it return errors or does it return out-of-date data?

## Examples

### HBase

Guarantees that each read will get the most recent write. However, if a region is unavailable read requests will fail.

### PostGres HA

PostGres in a leader/follower configuration provides availability but not consistency. The follower will often provide an out-of-date copy of data.

---

Tags: #Data
Reference: [CAP theorem (Wikipedia)](https://en.wikipedia.org/wiki/CAP_theorem)