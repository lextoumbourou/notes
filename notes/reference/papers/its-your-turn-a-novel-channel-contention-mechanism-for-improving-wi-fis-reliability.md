---
title: "It's Your Turn: A Novel Channel Contention Mechanism for Improving Wi-Fi’s Reliability"
date: 2024-10-14 00:00
modified: 2024-10-14 00:00
status: draft
---

These are my notes from the paper ["It's Your Turn": A Novel Channel Contention Mechanism for Improving Wi-Fi's Reliability](https://arxiv.org/abs/2410.07874) by Francesc Wilhelmi, Lorenzo Galati-Giordano and Gianluca Fontanesi. Note: some text from the paper is included unmodified.

## Overview

This research paper proposes a new wi-fi channel contention mechanism called [It's Your Turn](../../permanent/its-your-turn.md) (IYT), designed to improve the reliability of Wi-Fi networks.

The authors argue that the current [Listen-Before-Talk](../../permanent/listen-before-talk.md) (LBT) system, which relies on random backoff, leads to unreliable performance and high latency, especially in dense networks.

IYT tackles these issues by leveraging environmental awareness to establish an ordered access to the channel, based on neighbouring activity.

Through simulation results, the paper demonstrates that IYT achieves greater determinism and adaptability than existing backoff mechanisms like [Binary Exponential Backoff](../../permanent/binary-exponential-backoff.md) (BEB) and [Deterministic Backoff](../../permanent/deterministic-backoff.md) (DB), making it a suitable candidate for improving reliability in the upcoming Wi-Fi 8 standard.

## Algorithm

IYT is a backoff computation method that builds on top of the DCF, thus being compatible with the current IEEE 802.11 standard. IYT’s backoff computation is based on environmental awareness so that neighboring inter-BSS activity is considered for achieving an ordered access to the medium and, hence, achieving higher determinism.

The operations done by IYT are as follows:

1. **Neighboring activity detection**: Inter-BSS transmissions are overheard to identify the presence of overlapping BSSs (Alg. 1, line 3). Neighboring activity is done in practice by inspecting the headers of the detected inter-BSS frames (e.g., data frames, Beacon frames, Probe request/response), which include relevant information such as the source and destination address, or the BSS ID. Alternatively, the BSS Color, which is included in Physical Layer Convergence Protocol (PLCP) headers of 802.11ax and onwards frames, can be retrieved and used to quickly identify inter-BSS activity. In Alg. 1 (lines 10 − 12), the neighbor list is updated upon detecting strong enough inter-BSS signals, so that the power received at the device of interest j, P(j) is above the CCA threshold.
2. **Neighboring list ordering**: Based on the neighbor devices detected by j, an ordered list Lj is maintained, which determines the order in which channel accesses must be done. The list’s order can be determined, for instance, based on BSS ID or the BSS Color of the different involved BSSs, ordered in ascending order. Using such an order, a policy like RR can be adopted. As a way of example, L = {BSS2, BSS1, BSS3} would be obtained by any of the 3 BSSs using colors 0001, 0010, and 1011, respectively, for RR with ascending order. Notice that other ordering policies could be adopted, depending on traffic priorities, device conditions (e.g., devices with good channel conditions may access the channel more frequently), or any other aspects.
3. **Backoff computation**: The neighboring list is used to compute the backoff, BO, in a way that the established order Lj is respected. In particular, a token T indicating the device to transmit next is maintained by each device j to keep track of the status of the transmission order. The token is updated every time a device i from the list ends a transmission (Alg. 1, line 14), which allows computing the distance of the device of interest j to the token (Alg. 1, lines 15−18). The distance to the token d is used to compute the bounds [CWmin − 1, CWmax − 1] among which to sample the random backoff (Alg. 1,

## Abstract

The next generation of Wi-Fi, i.e., the IEEE 802.11bn (aka Wi-Fi 8), is not only expected to increase its performance and provide extended capabilities but also aims to offer a reliable service. Given that one of the main sources of unreliability in IEEE 802.11 stems from the current distributed channel access, which is based on Listen-Before-Talk (LBT), the development of novel contention schemes gains importance for Wi-Fi 8 and beyond. In this paper, we propose a new channel contention mechanism, “It’s Your Turn” (IYT), that extends the existing Distributed Coordination Function (DCF) and aims at improving the reliability of distributed LBT by providing ordered device transmissions thanks to neighboring activity awareness. Using simulation results, we show that our mechanism strives to provide reliable performance by controlling the channel access delay. We prove the versatility of IYT against different topologies, coexistence with legacy devices, and increasing network densities.

Index Terms—Beyond Listen-Before-Talk, IEEE 802.11, Wi-Fi

## Conclusion

In this paper, we proposed IYT, a novel backoff mechanism that improves the reliability of decentralized channel access in Wi-Fi. We evaluated IYT using the Komondor simulator and compared its performance against BEB and DB under full-buffer traffic conditions. Our results showed that IYT provides high determinism and adapts well to various situations, thus improving the worst-case performance compared to the other studied baselines (BEB and DB) and positioning itself as a strong candidate for improving reliability in Wi-Fi 8. We also showed some appealing properties of IYT regarding coexistence (it is legacy-friendly), scalability (it preserves determinism even when network density increases), and adapt-ability (it performs well in large, random deployments).
