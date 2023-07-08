---
title: The Bits and Bytes of Computer Networking
date: 2023-03-25 00:00
category: reference/moocs
slug: computer-networking
summary: "Notes from [The Bits and Bytes of Computer Networking](https://www.coursera.org/learn/computer-networking)"
status: draft
---

I took this course to obtain the [Google IT Support Professional Certificate](https://www.coursera.org/professional-certificates/google-it-support), which itself was taken as a Recognition Of Prior Learning credit for another course.

At the time of taking the course, I had previously learned all the concepts, however, many of them I have become quite out of practice with, so thought it'd be useful to add notes.

## Week 1

* [[TCP / IP]]
    * Transmission Control Protocol
    * Internet Protocol
* [[NAT]]
    * Network Address Translation.
* [[TCP/IP Five-Layer Network Model]]
    * [[Physical Layer]]
        * Physical devices like network cables, ports, cards etc
        * Includes specifications for the network cables.
    * [[Data Link Layer]]
        * Ethernet / Wi-Fi
        * Defines a common way of interpretting signals from physical so network devices can communicate.
        * Switches live here.
    * [[Network Layer]]
        * IP
        * Aka the internet layer.
        * Routers
    * [[Transport Layer]]
        * TCP/UDP
        * Figures out which client or server programs should get the data.
    * [[Application Layer]]
        * HTTP / SMTP etc
* Internetwork
    * A collection of networks.
    * Most famous: the internet.
* Border Gateway Protocol (BGP)
    * Routers share data with each other via this protocol.
    * Let's them find optimal paths to forward traffic.

## Week 2

* [[Subnetting]]
    * Address classes let us break total space of global IP addresses into smaller subnetworks.
    * The individual subnets have own gateway router's serving as ingress and egreess point.
* [[Subnet Masks]]
    * 32-bit numbers written out as four octets in decimal.
    * Without subnets, we could use some of those four bytes of ip to identify the network, and the others to identify the host.
    * Subnetting let's use kinda break it down even further:
      ![[it-support-subnet-example.png]]
    * Introduce a [[Subnet Id]]
        * Some of the bits can represent the subnet id.
    * At internet level, core routers only care about the network id.
        * They use it to send the datagram across to other gateway router for the network.
    * The gateway router may send to another router in the path using subnet info.
    * The last router uses to host id.
    * Example:
        * IP address: 9 . 100 . 100 . 100
        * y
        * Subnet mask in binary: 255 . 255 . 255. 0
        * The numbers in the octets that have a corresponding 1s in the column, after the parts defined by class, are the subnet id, the 0s are the host id.
        * A single 8 bit number can represents 256 numbers: 0-255, so we get 256 ips, and 2 are reserved. Leaving 254 available for assignment.
    * Example 2
        * A subnet mask: 255. 255 .255 .224
        * Submask in binary: 11111111 11111111 11111111 11100000
        * That leaves us with 5 0s for host ids and 27 for the network (32 all up)
        * Another way to write the subnet is: 9.100.100.100/27
* [[Basic Binary Math]]
    * Normal numbering system is base 10.
    * Binary is considered base 2.
    * To find number of representations you can fit in an N-bit binary number: 8-bit = 2^8 = 256
        * You can switch the base for different bases:
            * 10^2 = 2 columns of decimal digits (0 - 99)
            * 10^3 = 3 columns of decimal digits (0 - 999)
    * When adding with binary:
        * 0 + 0 = 0
        * 0 + 1 = 1
        * 1 + 0 = 1
        * 1 + 1 = 10 (carry the digit to the next column)
    * In logic, 1 represents true and 0 represents false.
    * Two important operators: OR and AND
        * X OR Y = Z
            * If either X or Y is true, then Z is true.
            * 1 OR 0 = 1
            * 1 OR 1 = 1
        * X AND Y = Z
            * If X and Y are true Z is true, else false.
            * 1 AND 1 = 1
            * 1 AND 0 = 0
            * 0 AND 0 = 0
    * A subnet mask is a way for a computer to use [AND (Boolean Operation)](../../../../permanent/boolean-operation-and.md) operators to determine if an IP address exists on the same network.
    * Example:
        * 00001001 . 0110 0100 . 01100100 . 01100100
        * AND
        * 1111 1111 . 1111 1111 1111 1111 . 0000 0000
        * =
        * 00001001 . 0110 0100 . 01100100 . 0000 0000
        * Result: 9.100.100.0
* [[CIDR]]
    * **Classless** Inter-Domain Routing
        * In other words, removes the Class A/B/C system.
    * As internet grew, traditional subnetting couldn't keep up.
    * 254 networks in a class C network is too small for most use cases.
    * Whereas 65,534 in a class B, is way too much.
    * Demarcation point
        * To describe where one network or system ends and another one begins
    * In previous model, network id, subnet id and host id were needed.
    * With CIDR, network and subnet id are combined into one.
    * Example:
        * 9.100.100.100/24
        * Describes a ip address: 9.100.100.100
        * Also the netmask: 255.255.255.0
    * Allows for arbitrary network sizes, can use /23 or /22.
* Routing
    * [[Router]]
        * Network device that forwards traffic depending on destination address.
        * Basic routing:
            * 1. Receive data packet.
            * 2. Examine destination IP.
            * 3. Look up IP destination network in routing table.
            * 4. Forwards traffic to destination (or closest place to network router knows about).
        * When router interface receives a packet for its network, it strips away the data link layer headers and footers.
            * It inspects the IP header to figure out which node to send it to.
            * Then creates a new data-link header with a new checksum value for the datagram.
                * Decrements the TTL field (TTL is usually 64).
            * Then sends to the destination mac address.
    * [[Routing Tables]]
        * Most basic routing table would have 4 columns:
            * Destination network - definition of network (one column if CIDR notation).
            * Next hop - where to forward traffic for this network.
            * Total hops - how many hops to get to the network.
            * Interface - which interface on the device to use.
    * [[Interior Gateway Protocols]]
        * [[Routing Protocols]]
            * Protocols routers use to share information with other routers.
            * Fall into 2 main categories:
                * Interior Gateway Protocols
                * Exterior Gateway Protocols
            * [[Interior Gateway Protocols]]
                * Used by routers to share info within a single [[Autonomous System]]
                    * Autonomous System: a collection of networks under the control of a single network operator.
                        * Ie a single corporation with multiple offices with their own LANs.
                * Interior can be split into 2 further categories:
                    * [[Distance-vector protocols]]
                        * An older standard.
                        * Has a routing table which includes every network known to it and how far away they are.
                        * Routers can send contents of their routing tables to each other, to share information about distance to nodes.
                    * [[Link state routing protocols]]
                        * Each routers shares info about the state of their interfaces:
                            * Interfaces can be direct connections to other routers or networks.
                        * Info about each router is propagated to every other router on autonomous system.
                        * Therefore, routers on the system knows the entire state of every other router.
                            * So each router can be decisions about the best path to each network.
                            * Requires more memory and compute to operate.
    * Exterior Gateways, Autonomous Systems and the IANA
        * Routers use [[Exterior Gateway Protocols]] when they need to share info across orgs.
        * The IANA or the Internet Assigned Numbers Authority is a nonprofit organization that helps manage things like IP address allocation
        * Along with managing IP address allocation, the IANA is also responsible for ASN or Autonomous System Number allocation
        * [[ASN]]
            * A number assigned to individual autonomous systems.
            * Like IPs, they're 32 bit numbers, but they are referred to as a a single decimal number, not split into readable bits.
            * For example, AS19604 is the ASN assigned to IBM.
        * Their is only one exterior gateway protocol in use today: [[Border Gateway Protocol]]
    * Non-Routable Address Space
        * IP is a single 32-bit number.
        * Single 32-bit number can represent 4,294,967,295 unique numbers.
        * Not enough for every person on the planet, let alone data centers, phones etc.
        * In 1996, RFC 1918 introduce non-routable address space: which are ranges of IPs that cannot be routed too (ie internal networks).
            * Defines 3 ranges of IPs in this space
                * 10.0.0.0/8
                * 172.16.0.0/12
                * 192.168.0.0/16
        * This is not NAT - that comes later.

## Week 3

### The Transport Layer

* [[Transport Layer]]
    * Responsibilities:
        * Multiplexing and demultiplexing.
        * Establishing long running connections.
        * Data integrity through error checking and data verification.
    * [[Multiplexing]]
        * Nodes on a network can deliver traffic toward many receiving services.
    * [[Demultiplexing]]
        * Taking traffic aimed at the same node and delivering to proper receiving service.
    * [[Port]]
        * 16-bit number used to direct traffic to specific services on a networked device.
        * Ports allow a single server to host many networked applications.
        * Common ports:
            * 80 - HTTP
            * 443 - HTTPS
            * 21 - FTP
            * 25 - SMTP
* Dissection of a TCP Segment
    * TCP segment: header and data section.
    * Header:
        * Source port - high-numbered port chosen from special section of ports called ephemeral ports.
        * Destination port - port of service traffic is going to.
        * Sequence number - 32-bit number that tracks where in sequence this one lives.
            * At transport layer, data is split into many segments.
        * Acknowledgement number - next expected segment.
        * Data offset field
            * 4-bit number that tells you how long the TCP header is.
            * 6-bits reserved for TCP control flags.
            * TCP window - specifies the range of sequence numbers that might be sent before an ack is required.
            * Checksum
                * Same as checksum at IP and ethernet level.
                * Checksum is calculated across entire segment.
            * Urgent point field
                * Point out if particular segments are more important than others.
                * Feature of TCP that is rarely used.
            * Options field
                * Rarely used in real world.
            * Padding
                * Sequence of 0s to ensure payload starts at expected point.
    * Ethernet frame encapsulates an IP datagram in the payload section.
    * Datagram encapsulates a TCP segment in payload section.
* TCP Control Flags and the Three-Way Handshake
    * TCP uses control flags to establish connections.
        * [[URG]]
            * Urgent.
            * 1 = segment is considered urgent and urgent fieldh as more info.
        * [[ACK]]
            * Acknowledged.
            * 1 = means that acknowledgement number should be examined.
        * [[PSH]]
            * Push
            * 1 = transmitting device wants receiving device to push buffered data to application receiving end.
                * buffer means that certain data is held somewhere before being sent somewhere else.
        * [[RST]]
            * Reset.
            * One of sides in connection hasn't recovered correctly and needs to start from scratch.
        * [[SYN]]
            * Synchronise.
            * used when first establishing TCP connection
                * makes sure receiving end knows to examine sequence number field
        * [[FIN]]
            * Finish
            * When 1, transmitting computer has no more data to send.
* [[Three-Way Handshake]]
    * Computer A send TCP segment to Computer B with a SYN flag sent to establish connection.
    * Computer B responds with a TCP segment where both SYN and ACK are sent to acknowledge.
    * Computer A send ACK flag, which is an acknowledgement. Then starts sending data.
    * Happens every time a TCP connection is established.
    * To close the connection:
        * Computer B send FIN flag.
        * Computer A send ACK.
        * Computer A send FIN.
        * Computer B send ACK.
* TCP Socket States
    * Instantiation of an endpoint in a potential TCP connection.
    * You can send data to any port, but you will only get a response if the server has opened a socket.
    * TCP sockets can be in lots of states:
        * LISTEN - socket is listening and ready for incoming connections.
            * See this server side only.
        * SYN_SENT - synchronisation request has been sent, but connection hasn't been established.
            * Client-side only.
        * SYN_RECEIVED - socket previously in LISTEN state but recieved a synchronisation request and sent a SYN/ACK back, but hasn't got final ACK.
            * Server-side only.
        * ESTABLISHED - TCP connection is working order and both sides are free to send data.
            * Server and client-side.
        * FIN_WAIT - FIN has been sent but no ACK from other side.
        * CLOSE_WAIT - connection closed at TCP layer but application hasn't released hold on socket yet.
        * CLOSED - connection fully terminated and no further comms possible.
    * Other socket states exist, and are different from OS to OS.
* Connection-oriented and Connectionless protocols
    * TCP is a connection-oriented protocol
        * They protect against network failures usign connections, acks and checksums.
        * Since TCP sends ACK for every data is sees, it is in good position to know if something failed.
        * Sequence numbers allows for resending parts of data.
        * But this is a lot of overhead.
    * Connectionless
        * Most common is [[UDP]]
        * Common use case is video streaming: if you lose a few frames, it's not the end of the world.
* Categories of ports:
    * [[System Ports]]
        * 1 - 1023. Common apps like FTP, Telnet
    * [[User ports]]
        * 1024 - 49151
    * [[Ephemeral Ports]]
        * 49152 - 55536
        * Temporary ports for transfer.
* Firewalls
    * A device that blocks traffic that meets certain criteria.
    * Can operate at different levels.
        * Most commonly at Transport level (port based)

### The Application Layer

* The payload section is the data that the applications are actually trying to send.
* [[OSI Model]]
    * A 7 layer networking model.
        * Application
        * Presentation
        * Session - responsible for handling connection between applications
        * Transport
        * Network
        * Data link
        * Physical
    * Often used in academic settings.

## Week 4

* DNS
    * Domain Name System
* Name Resolution
    * Process of turning domain name into an IP address
    * 5 primary types of DNS servers
        * Caching name servers
        * Recursive name servers
            * Usually combined with caching name server.
            * Check cache, then perform recursive lookup.
        * Root name servers
            * 13 total.
            * They direct queries to appropriate TLD name server.
            * Distributed across the globe using [[AnyCast]].
            * Not actual servers, more services.
        * TLD name servers
            * Last past or any domain name (.com part of url)
            * For each TLD, there's a name server (or collection of servers - ie service)
        * Authoritative name servers
            * Responsible for the last 2 parts of the domain name.
    * TTL
        * How long name server should cache entry before discarding it.
        * Used to be really long - 1 day - due to having limited bandwidth.
        * Shorter now.
* [[AnyCast]]
    * A technique used to route traffic to different destinations based on factors like location, congestion or link health
* [[DNS and UDP]]
    * DNS uses UDP instead of TCP.
    * Why?
        * A single response can fit inside a single UDP protocol.
        * DNS needs to make a lot of calls between servers, TCP has too much packet overhead.
            * 44 packets would be required to make a single recursive DNS lookup with TCP.
            * Only 8 for UDP.
        * DNS server handles error recovery at the application level.
            * It just asks again if it doesn't get a response.
        * DNS server only has to response to incoming lookups.
    * DNS does happen over TCP in some cases.
        * If a response is too large for a single packet, DNS will respond with too large, and request connection via TCP.
* Resource Record Types
    * A record
        * Point a domain name to a specific IPv4 address.
        * A single domain name can have multiple records. It returns IP addresses in round robin.
    * Quad A
        * Like A record but returns IPv6 record
    * CNAME record
        * Alias record - direct traffic from one domain to another.
    * MX record
        * Mail exchange - used to route to mail server.
    * SRV record
        * Service record
        * Define the location of specfic services (CalDav)
    * TXT record
        * Originally used as a sort of comment for DNS, now it's used to pass messages between computers.
    * NS record
        * Servers responsible for a zone.
    * PTR
        * Resolves an ip to name.
* Anatomy of a Domain Name
    * FQDN - fully-qualified domain name (the entire domain).
        * Max characters is 255.
    * Registrar
        * Company that has an agreement with [[ICANN]] to sell domain names.
* DNS Zones
    * Hierarchical concept.
    * Each authoritative name server is responsible for their zone.
    * Reverse lookup zone files
        * Convert ips to names.
* DHCP
    * Dynamic host configuration protocol
    * Automates configuration of hosts on a network.
    * Runs at application layer of network.
    * Dynamic allocation
        * A range of IPs is set aside for devices.
    * Automatic allocation
        * Main diff is that DHCP keeps track of assignment to continue to give same ips out.
    * Fixed allocation
        * A manual list of MACs to IPs.
* DHCP in action
    * [[DHCP Discovery]]
        * Four step process:
            * 1. DHCPDISCOVER
                * DHCP client sends a DHCP discover message.
                    * DHCP always listens on port 67.
                    * Source port is 68.
                    * Encapsulated in IP datagram with source 0.0.0.0:68 and dest: 255.255.255.255:67
                * Broadcast message sent to every machine on LAN.
                    * If DHCP is present, receives message.
            * 2. DHCPOFFER
                * DHCP server checks its config, and figures out what to do.
                * Then sends DHCPOFFER message back to client.
                * Sent as a broadcast to every client in network.
                    * Includes client Mac address, so client knows what's up.
            * 3. DHCPREQUEST
                * Client confirms that it wants the IP from the server.
            * 4. DHCPACK
                * DHCP server confirms network configuration.
    * Configuration is called [[DHCP Lease]].
        * When lease expires, client needs to request another configuration.
        * Also happens when client disconnects.
* [[Network Address Translation]]
    * No specific standards, vendors implement them differently.
    * [[IP Masquerading]]
        * Router rewrites IP address field when passing through internal network, masking the true IP.
* NAT and the Transport Layer
    * When considering return traffic, one way that a
    * With one-to-many NAT, we've talked about how hundreds, even thousands of computers can all have their outbound traffic translated via NAT to a single IP.
    * NAT rewrites the IPs of internal nodes to a single IP.
    * But how does it deal with return traffic to the IPs?
        * [[Port Preservation]]
            * Keep track of the original source by opening the same port on the router.
        * [[Port Forwarding]]
            * An approach where a request for a certain port is always routed to a certain node.
* [[VPN]]
    * A technology that allows for extension of private or local network to hosts that might not work on the same local network.
    * A tunnelling protocol.
        * Employees create a "tunnel" to their network.
    * They mostly work using the payload section of the transport layer to carry encrypted payload
        * It contains an entire second set of:
            * packets
            * the network
            * the transport
            * the application layers
* Proxy services
    * A server that acts on behalf of the client to provide additional functionality
        * Anonymity.
        * Security.
        * Content filtering.
        * Increased performance.
    * Gateway router is an example of a proxy (though rarely referred to this way).
    * Common examples of proxies:
        * web proxies
            * useful to speed up internet when the internet was slower than it was today.
            * might used for content filtering to block traffic to certain sites.
            * also used for dealing with decryption.

## Week 5

* POTS and Dial-up
    * [[Public Switched Telephone Network (PSTN)]]
        * Sometimes referred to as plain-old telephone system (or POTS)
    * Students from Duke University figured out that you can exchange data over the telephone network.
        * They created a digital bulletin system Usenet.
    * [[../../../../journal/ai-art/stable-diffusion/models]]
        * Stands for modulator demodulator
        * Take data and turn them into audible wavelengths to be transmitted over phone lines.
        * Similar to how line coding turns one and zeros into modulating electrial chages across Ethernet cables.
    * [[Baud Rate]]
        * A a measurement of how many bits per second can be passed alone a phone line.
        * 110 p/s was about the rate in the late 1950s.
        * By the time Usenet was developed, it's more like 300 bits.
        * By early 90s, it was around 14.4 kilo bits per second.
* [[Broadband]]
    * Few definitions, but usually means anything that isn't dial-up.
    * Refers to connections that are always on.
    * [[T-Carrier Technology]]
        * Allowed lots of phone calls to travel along a single cable.
        * Invented by AT&T.
        * Transmission System 1 or T1 for short.
        * Can carry 24 calls across a single pair copper.
        * Later pepurposed for data transfer.
        * Original speeds capable of 1.544 megabits per second.
        * Further improved to allow multiple T1s via single link
        * AT3 line is 28 T1s all multiplexed, achieving a total throughput speed of 44.736 megabits per second.
        * Surpassed by modern broadband.
    * [[Digital Subscriber Lines]]
        * Telephone companies found that they can transmit more data on copper lines than needed for voice.
        * By using freq that voice didn't use, they could send more data *and* not interfere with voices.
        * DSL uses their own modems, known as DSLAMs or digital subscribed line access multiplexers.
            * Connection is established when DSLAM is powered on, and torned down when turned off, unlike dial-up.
        * Most 2 common types of DSL:
            * [[ADSL]]
                * Asymmetric Digital Subscriber Line.
                    * ADSL connections have diff speeds for incoming and outbound data
            * [[SDSL]]
                * Symmetric Digital Sub Line.
                * Downloads and uploads are the same bandwidth.
                * Most have upper cap as 1.544 megabits per second.
                * Further development of SDSL: HDSL that can beyond upper cap.
* [[Cable Broadband]]
    * First cable TV was developed in 1940s.
        * Used to provide TV to remote towns out of range of TV tower.
    * Cable television expanded slowly until 1984 when the Cable Communications Policy Act was passed.
        * Deregulated cable and caused a boom.
    * Cable providers soon found that they could use their coax technology to deliver data on top of TV.
    * Cable internet uses shared bandwidth technology, users in block or subdivision in suburb share bandwidth.
        * Can impact performance at bust times.
        * Although mostly cable operators update networks to avoid that happening.
    * Connections managed by [[Cable model]]
        * Device that lives at the edge of consumer network to connect to cable modem termination system or CMTS
* [[Fiber Connections]]
    * Core of internet uses fibre for connections for a long time.
        * Allows for faster speeds without degredation.
    * Maximum distance an electrical signal can travel across a copper cable before it degrades too much and requires a repeater is thousands of feet.
        * Fiber connections can travel many miles before signal degrades.
    * More expensive than using copper cables
    * [[FTTX ]]
        * Stands for fiber to the x
        * FTTN - fibre to the neighbourhood.
        * FTTB - fibre to the building.
        * FTTH - fibre to the home.
        * FTTP - fibre to premise. Either FTTH or FTTB.
    * [[ONT]]
        * Cable doesn't use modems, so demarcation point known as optical network terminator or ONT.
        * ONT converts data from fiber network protocols to be compatible with twisted pair copper networks.
* [[Point to Point Protocol (PPP)]]
    * byte-oriented protocol broadly used for high-traffic data transmissions
    * At the Data Link layer to trasnit data between 2 devices on same network.
    * Options:
        * Multilink - method for spreading traffic across multiple distinct PPP connections.
        * Compression - reduce data in frame.
        * Auth
            * Passwrod Auth Protocol (PAP)
                * Password auth option
            * Challenge Handshake Auth Protocol (CHAP)
                * 3-way handshake.
        * Error detection - includes Frame Check Sequence (FCS) and looped link detection.
            * Frame Check Sequence (FCS)
            * Looped link detection
    * Sub-protocols for PPP
        * Network Control Protocol (NCP)
        * Link Control Protocol (LCP)
    * Data is sent in [[PPP Frame]].
        * File format has the following fields:
            * Flag - single byte to mark beginning of the frame.
            * Address is a single byte for broadcast address.
            * Control is a single byte required for various purposes but also allows a connectionless data link.
            * Protocol varies from one to three bytes which identify the network protocol of the datagram.
            * Data is where the information you need to transmit is stored and has a limit of 1500 bytes per frame.
            * Frame check sequence (FCS) is 2 or 4 bytes and is used to verify data is intact upon receipt at the endpoint.
    * Encapsulation
        * Process by which each layer takes data from previous layer and adds headers and trailers for next layer to interpret.
        * When sent to endpoint, process is reversed de-encapsulation.
    * PPP can be hard to manage with direct links required, so PPP over Ethernet was invented.
    * [[Point to Point Protocol over Ethernet (PPPoE)]]
        * A method for encapsulating PPP frames inside an ethernet frame.
            * Tunnels packets over the DSL connection service provider's IP network and from there to the rest of the Internet
        * A common use case is PPPoE using DSL services where a PPPoE modem-router connects to the DSL service or when a PPPoE DSL modem is connected to a PPPoE-only router using an Ethernet cable.
        * PPP is strictly point-to-point, so frames can only go to the intended destination.
        * PPPoE requires a new step because ethernet connections are multi-access enabled (every node connects to another).
            * Adds extra step called the discovery stage.
                * The discovery stage establishes a session ID to identify the hardware address.
                * This stage ensures data gets routed to the correct place.
* [[Wide Area Network Technologies]]
    * Used when you need office to office communication over the internet.
    * Area between each demarcation point and the ISP's actual core network is called a [[Local Loop]]
    * Physical versus software-based WANs
        * [[WAN router]]
            * Aka [[Border Routers]] or [[Edge Routers]].
            * Hardware devices that act as intermediate systems to route data amongst the LAN member groups of a WAN (also called WAN endpoints) using a private connection.
            * Facilitate an organization’s access to a carrier network.
            * Include a digital modem interface for the WAN, which works at the OSI link layer, and an Ethernet interface for the LAN.
        * Software-Defined WAN (SD-WAN)
            * Software developed to address the unique needs of cloud-based WAN environments.
            * SD-WANs can be used alone or in conjunction with a traditional WAN.
            * SD-WANs simplify how WANs are implemented, managed, and maintained.
            * An organization’s overall cost to operate a cloud-based SD-WAN is significantly less than the overall cost of equipping and maintaining a traditional WAN.
            * One of the ways that SD-WANs help reduce operational costs is by replacing the need for expensive lines leased from an ISP by linking regional LANs together to build a WAN.
        * WAN optimisation
            * Compression
                * Reducing file size to improve network efficiency.
            * Deduplication
                * Prevents files from storing multiple times to avoid wasting space.
            * Protocol Optimization
                * improve efficiency of network protocolocs for apps that need higher bandwidth and low latency.
            * Local caching
                * Store local copeies of network and internet files.
            * Traffic shaping
                * Optimizing network performance by controlling the flow of network traffic. Three techniques are commonly used in traffic shaping:
                    * bandwidth throttling - controlling network traffic volume during peak use times
                    * rate limiting - capping maximum data rates/speeds
                    * use of complex algorithms - classifying and prioritizing data to give preference to traffic.P
        * WAN protocols
            * Used in conjunction with WAN routers to perform task of distinguishing between private LAN and related public WAN
            * WAN protocols
                * Packet switching
                    * A method of data tranmission where packets are broken into multiple packets.
                    * Each packet has header explaining how to replicate.
                    * Packets are triplicated to prevent data corruption.
                * Frame relay
                * Async Transfer Mode (ATM)
                * High level data control (HLDC)
                * Packet over Sync Optical Network (SONET)
                * Multipprotocol Label Sitching (MPLS)
* [[Point-to-Point VPNs]]
    * A point to point VPN also called a site to site VPN establishes a VPN tunnel between two sites. This operates a lot like the way that a traditional VPN setup lets individual users act as if they're on the network they're connecting to. It's just that the VPN tunneling logic is handled by network devices at either side so that users don't all have to establish their own connections.
* [[Introduction to Wireless Networking Technologies]]
    * Most common spec for how wireless devices communicate is defined by IEEEE 802.11 standards.
        * Aka 802.11 family.
        * Make up technology called: Wi-Fi.
    * Communicate with each other through radio waves.
        * Different 802.11 standards use the same basic protocol, but might operate at diff frequency bands.
    * [[Frequency Band]]
        * A certain section of the radio spectrum that's been agreed upon to be used for certain communications.
    * [[FM Broadcast Band]]
        * A specific frequency band.
        * In North America, between 88 and 108 megahertz.
    * Wi-Fi networks operate on a few different frequency bands:
        * Commonly the 2.4 gigahertz and 5 gigahertz bands.
    * Common 802.11 specifications in order of adoption: 802.11b, 802.11a, 802.11g, 802.11n and 802.11ac
        * Improvements are usually higher speeds or more simulatenous users.
    * 802.11 protocol define both the physical and the data link layers.
    * [[Wireless Access Point]]
            * A device that briges the wireless and wired portions of a network.
    * 802.11 frame has a number of fields:
        * First 2 octets = frame control field.
            * 16-bits long
            * Contains that fields that can describe how to process.
        * Duration field
            * How long total frame is
            * Receiver knows how long it should expect to listen to transmission
        * 4 address Fields (MAC addresses - 6 bytes):
            * Source address field
            * Intended destination
            * Receiving address
                * The MAC address of the access point that should receive the frame.
            * Trasmitter address
                * The MAC address of whatever has just transmitted the frame.
        * Sequence control field
            * 16-bit long
            * Between 3rd and 4th address. Contains a sequence number of keep track of ordering the frames.
        * Data payload
            * Payload of the protocols further up the stack.
        * Frame check sequence field
            * Contains a checksum for cyclical redundancy check, like how ethernet does it.
* [[Wi-Fi 6]]
    * Formely known as 802.11ax
    * One of the biggest improvements in Wi-Fi tech:
        * Higher data rates
        * Increased band capacity.
        * Better performanced
        * Improved power efficiency.
    * Capabilities:
        * Channel sharing
        * Target Wake Time (TWT)
            * allow battery-powered devices to sleep when not in use.
        * Multi-user MMO (Multiple Input, Multiple Output)
        * 160 MHz channel utilisation.
        * 1024 Quadrature amplitude modulation
            * Combines 2 signals into a single channel.
        * Orthogonal Frequency Division Mutltiple Access (OFDMA)
        * Transmit beamforming.
            * Allow for more efficient higher data rates by targeting each connected device.
    * Wi-Fi 6E extends Wi-Fi 6 into 6 GHz
        * Additional certification for Wi-Fi 6 that adds a 3rd 6 Ghz band.
        * Wi-Fi 6E has more channels to use to broadcast: includes 14 more 80MHz channels and 7 more 160Mhz channels.
* [[Wi-Fi Standards]]
    * Many wireless tech that uses various frequencies: Wi-Fi, Z-Wave, ZigBee, Thread, Bluetooth, and Near Field Communication (NFC).
    * Radio and microwave frequency bands each have specific ranges that are divided into channels.
    * Wi-Fi uses 2.4 GHz and 5 GHz microwave radio frequency band ranges.
    * Some Wi-Fi routers use multiple channels within each range to avoid signal interference and to load-balance network traffic.
    * Wi-Fi is commonly used for wireless local area networks (WLANs).
    * 2.4 GHz
        * Has longest signal range at 150 feet (45m) indoors to 300 feet (92m) outdoors.
            * Risk of being intercepted by crims.
        * Can pass through solid objects.
        * Limited number of channels.
        * Can experience network traffic congestion and interferece
            * Bluetooth can overlap for example
            * Microwave ovens can overlap.
        * Map data rate is 600 Mbps.
    * 5 Ghz
        * Way more channels than 2.4 Ghz.
        * Fewer interference problems and less wireless network traffic.
        * Achieves 2 Gbps data transfer speeds.
        * Wireless range is limited to 50 feet (12 meters) indoors and 100 feet (30 meters) outdoors.
        * Does not penetrate walls and other solid objects as well as 2.4 Ghz.
* IEEE 802.11 standards
    * IEEE created first 802.11 standard in 1997.
    * IEEE will keep updating the spec until a new tech replaces Wi-Fi.
    * When devices are configured to access Wireless Access Point, it's considerd "infrastructure mode".
    * 802.11 specifications use the same fundamental data link protocol, differences may be at:
        * signal ranges
        * modulation techniques
        * transmission bit rates
        * frequency bands
        * channels
    * Countries can impose different regulations on channel usage, power limitations, and Wi-Fi ranges.
    * [[Dynamic Frequency Selection (DFS)]]
        * A technology required to prevent 5 GHz Wi-Fi signals from interfering with local radar and satellite communications.
* Wireless Network Configurations
    * Ad-hoc networks
        * Simplest possible network.
        * Nodes all speak directly.
        * Used when other network communication is down.
    * Wireless LANs or WLANs
        * Most common wireless network in business world.
        * One or more access points. Each accses point is a bridge between wired and wireless networks.
    * Mesh networks
        * Similar to ad-hoc networks, lots of devices communicate with each other forming a mesh.
        * Mostly, you're find mesh networks are made up of WAPs, that each communicate on a wired network.
* [[Wireless Channels]]
    * Individual, smaller sections of the overall frequency band used by a wireless network.
    * Addresses the problem of [[Collision Domains]]
        * On wired network, switches have mostly addressed this problem.
    * 2.4Ghz band is actually 2.4Ghz to 2.5Ghz
        * Between these is a series of channels that can be used.
        * Each country has slightly different standards.
    * Most WAPs perform analysis of which channels are most congested and adjust accordingly.
* [[Wireless Security]]
    * Wired links give you some inherent privacy that you don't get with wireless.
    * [[WEP]]
        * A standard to fix this.
        * Wired Equivalent Privacy.
        * Only provides a low-level of security: same as sending unencrypted data on wired network.
        * Only uses 40 bits for encryption key.
        * Replaced by WPA.
    * [[WPA]]
        * Uses 128-bit key.
        * Surprased by WPA2
    * [[WPA2]]
        * Uses 256-bit key.
    * Also using MAC address filtering is a good idea to prevent rouge devices on network.
* [[WPA3]]
    * Two versions: WPA3-Personal and WPA3-Enterprise
    * WPA3-Personal
        * Natural password selection
        * Ease of use
        * Forward secrecy - if passwrod is stolen, can still protect data
        * Simultaneous Auth of Equals (SAE)
            * reduces prob of dict attack.
    * WPA3-Enterprise
        * Galois/Counter Mode Protocol (GCMP-256)
        * Opportunistic Wireless Encryption (OWE)
        * Wi-Fi Device Provisioning Protocol (DPP)
        * 384-bit Hashed Message Authentication Mode (HMAC) with Secure Hash Algorithm (SHA)
        * Elliptic Curve Diffie-Hellman Exchange (ECDHE) and Elliptic Curve Digital Signature Algorithm (ECDSA)
* Cellular Networking
    * Aka mobile network
    * In some countries, cellular networks are most common way of connecting.
    * Just like wi-fi, operates over radio band.
    * Works over longer distances.
* Mobile Device Networks
    * Wireless networking works by sending a radio signal between two antennas
        * Might be printed on a circuit board or it might have a wire or ribbon that runs through your device

## Week 6

### Introduction to Troubleshooting and the Future of Networking

* [[Error Detection]]
    * Ability for protocol or program to determine if something went wrong.
* [[Error Recovery]]
    * Ability for protocol or program to attempt to fix an error.
* [[Internet Control Message Protocol]]
    * Mainly used by router or remote host to communicate why a transmission failed.
    * ICMP packet:
        * Header with a few fields.
            * Type field
                * Specifies what type of message is being sent.
            * Code field
                * Gives more info about the specific failure reason.
            * Checksum
                * 16-bit field.
            * Rest of header section.
                * Used for custom field.
        * Data section
            * Used by a host to figure out which of their transmissions generated the error.
    * ICMP wasn't designed for people to use, but two useful tools are that utilise ICMP are:
        * Ping
        * Traceroute
* [[Ping]]
    * Sends a special type of [[Internet Control Message Protocol]] called [[Echo Request]].
    * [[Echo Request]]
        * Includes just a destination.
        * Message equivalent to: "Hey, are you there?"
        * If destination is up, it will send back an ICMP message in reply, called an ICMP [[Echo Reply]] message.
    * Usage: `ping $destination_address`
        * Usually outputs how long the round trip took, TTL remaining and how large ICMP request is in bytes.
* [[Traceroute]]
    * Since communicates across network cross many intermediary nodes.
    * Traceroute can be used to see paths between two nodes.
    * Works using a manipulation of TTL field at IP level:
        * We know that TTL Field is decrememted by one at each router that forwards packet.
        * Traceroute usses TTL field by setting 1 for first pack, then 2 for second and so on.
        * By doing this, traceroute ensures first packet will be discarded by first router, resulting in time exceeded message. The 2nd packet, the 2nd router and so on.
    * Output:
        * Host (if resolvable) and IP at each hop.
        * Time in ms it took.
    * By default, Windows traceroute uses ICMP echo request.
    * Linux and macOS use UDP packets to very high port numbers.
    * Similar tools:
        * mtr on Linux
        * pathping on Window
* [[Netcat]]
    * At the command-line: `nc`
    * 2 mandatory arguments: host and port.
    * -v flag gives verbose info.
    * `-z` zero output mode.
    * [[Test-NetConnection]]
        * A Windows tool with similar functionality to [[Netcat]].

### Digging into DNS

* [[nslookup]]
    * Most common tool for troubleshooting name resolution issues.
    * Basic example: `nslookup google.com`:

    ```
    nslookup google.com
    Server:		192.168.86.1
    Address:	192.168.86.1#53
    
    Non-authoritative answer:
    Name:	google.com
    Address: 142.250.204.14
    ```

    * Includes interactive mode:

    ```
    nslookup
    > google.com
    Server:		192.168.86.1
    Address:	192.168.86.1#53
    
    Non-authoritative answer:
    Name:	google.com
    Address: 142.251.221.78
    ```

    * `set debug` command can be used to increase verbosity:

    ```
    > set debug
    > google.com
    Server:		192.168.86.1
    Address:	192.168.86.1#53
    
    ------------
        QUESTIONS:
    	google.com, type = A, class = IN
        ANSWERS:
        ->  google.com
    	internet address = 142.251.221.78
    	ttl = 96
        AUTHORITY RECORDS:
        ADDITIONAL RECORDS:
    ------------
    Non-authoritative answer:
    Name:	google.com
    Address: 142.251.221.78
    ```

    * Contain details like:
        * the TTL left, if it's a cached response
        * the serial number of the zone file the request was made against.
    * Returns `A` records by default, but can be configured to return other types of records.
* [[Public DNS Servers]]
    * ISP will nearly always give you acess to a recursive name server, and for most people this is all they're need for internet access.
    * Most business run their own DNS sserver for resolving internal hosts.
        * Useful to have backup DNS server.
    * Some companies run public DNS servers:
        * 4.2.2.1 is common
        * 4.2.2.6 is also common.
        * 8.8.8.8 is a DNS server provided by Google and officially documented.
    * Always do your research: a bad DNS server can hijack outbound DNS requests.
* [[DNS Registrar]]
        * A company that has an agreement with [[ICANN]] to sell domains.
        * Was originally just one company: [Network Solutions Inc](https://en.wikipedia.org/wiki/Network_Solutions) that operated entire .com space.
            * Later expanded to other companies (appeared NSI went wholesale).
        * Once you have a domain name, you can have configure your own name servers to be authoritive.
        * Domain names can be transferred from a registrar to another, usually using unique strings in TXT records.
        * Domains are leased for some duration. After that, anyone can take over the domain name if it isn't renewed.
* [[Hosts File]]
    * Original method of aliasing ip addresses with hostnames.
    * Each line contains an ip and the corresponding host name.
    * The reason that they still exists is for the loopback host.
    * Examined before DNS resolution occurs, so can be useful troubleshooting.
* [[Loopback Host]]
    * A way of sending traffic to yourself.
    * For ipv4: 127.0.0.1
* [[IPv6]]
    * A 32-bit size of an ip address was chosen, but this has not given us enough to support the explosion of the internet: IPv6 is a protocol to deal with this.
    * 128-bit binary number.
        * Two to the power of 128 would produce a 39-digit long number.
        * Supports an undecillion of ip address: roughly enough for every atom on earth.
    * Written as 8 groups of 16-bits
    * Example: 2001:0db8:0000:0000:0000:ff00:0012:3456
    * Has a notation method that can break it down further.
        * Every ip address that begins with 2001:0db8 has been reserved for documentation and training.
        * 2 rules for shortening:
            * Can remove any leading 0s from a group.
                * 2001:0db8:0000:0000:0000:ff00:0012:3456 -> 2001:db8:0:0:0:ff00:12:3456
            * Any consecutive group of 0s can be replaced with 2 colons.
                * 2001:db8:0:0:0:ff00:12:3456 -> 2001:db8::ff00:12:3456
    * Loopback address is 31 0s with a 1 at the end.
        * 0000:0000:0000:0000:0000:0000:0000:0001 -> ::1
    * Any address that begins with ff00 is for [[Multicast]]
        * A way of addressing groups of hosts at once.
    * Addresses beginning with FE80:: are used for [[Link Local unicast address]]
        * Allow for local network segment communication
        * Configured based on Mac address.
        * Similar to how DHCP works.
    * IPv6 allocated first 64 bit for network id, and 2nd 64 bits for host id.
    * You can still split network up for admin purposes: uses same [[CIDR]] notation as IPv4.
* [[IPv6 Headers]]
    * 1. [[Version Field]]
        * First field in IPv6 header is a Version field.
        * 4-bit field that defines what version of an IP is in use.
    * 2. [[Traffic Class Field]]
        * An 8-bit field that defines the type of traffic contained in the IP datagram
        * Allows for different classes of traffic to receive diff priorities.
    * 3. [[Flow Label Field]]
        * A 20-bit field used in conjuction wth Traffic class field to make decisions about quality of service level for specific datagram.
    * 4. [[Payload Length Field]]
        * 16-bit field that defines how long data payload section is.
    * 5. [[Next Header]]
        * Since IPv6 addresses are 4x as long as IPv4, they aim to keep header payload small.
        * To do that, the next header field allows for a chain of extra optional fields. Each optional field includes its own Next Header field.
    * 6. [[Hop Limit Field]]
        * 8-bit field identical in purpose to TTL field in IPv4 header.
    * 6. Source Address
    * 7. Destination Address.
    * If the next header field specified another header, it would follow. If not, a data payload, the same length as specified in the payload length field is next.
* IPv6 and IPv4 together
    * IPv6 needs to work alongside IPv4 to allow for gradual migration.
    * [[IPv4 Mapped Address Space]]
        * IPv6 specification sets aside IPv6 addresses that can be correlated to IPv4 address.
        * Any address that begins with 80 zeros, followed by 16 ones is part of it.
        * 192.168.1.1 == 0:0:0:0:0:fff:c0a8:0101
        * This let's IPv4 traffic to travel over IPv6 network.
    * [[IPv6 Tunnels]]
        * Consist of an IPv6 tunnel server on either end of a connection.
        * Take incoming IPv6 traffic and encapsulate in IPv4 datagrams.
        * IPv6 tunnel server deencapsulates and passes it along network.
        * Types of tunnels:
            * [[6in4/manual protocol]]
                * Encapsulates IPv6 packets inside an IPv4 packet
                * No additional headers to configure the setup of the tunnel endpoints
                * This protocol often will not function if the host uses network address translation (NAT) technology to map its IPv4 address.
            * [[Tunnel Setup Protocol (TSP)]]
                * Specifies rules for negotiating the setup parameters between tunnel endpoints.
                * This allows for a variety of tunnel encapsulation methods and wider deployment than is possible with the 6in4/manual protocol.
            * [[Anything in Anything (AYIYA)]]
                * protocol defines a method for encapsulating any protocol in any other protocol.
                * AYIYA was developed for tunnel brokers, a service which provides a network tunnel.
                    * Specifies the encapsulation, identification, checksum, security, and management operations that can be used once the tunnel is established.
                    * A key advantage: provides a stable tunnel through an IPv4 NAT.
                    * It allows users behind a NAT or a dynamic address to maintain connectivity even when roaming between networks.
    * [[IPv6 Broker]]
        * Companies that provide IPv6 tunnelling endpoints.
