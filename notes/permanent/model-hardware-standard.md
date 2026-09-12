---
title: Model Hardware Standard
date: 2026-09-02 06:27
modified: 2026-09-02 06:27
category: news
status: draft
summary: "Anthropic's research-preview standard aims to let AI agents discover, understand and safely operate programmable lab and manufacturing equipment."
tags:
  - AIAgents
  - Robotics
---

The **Model Hardware Standard (MHS)** is Anthropic's research-preview specification for letting AI agents operate physical equipment: microscopes, liquid handlers, robotic arms, cameras, lasers and manufacturing instruments.

The central idea is pleasantly unglamorous: standardise the driver layer. Today, each instrument tends to expose its own API and its own vocabulary, so integrating a lab or factory can take weeks or months of bespoke work. MHS gives a device a shared interface, makes it discoverable on a network, and describes both what it can do and the safety limits it must respect.

## How it works

An MHS driver exposes simple operations such as reading a temperature or writing a set point. It also carries natural-language metadata about the machine, including characteristics that code alone cannot convey, such as a robot arm's weight or operating limits.

An agent can use MHS through MCP, a command-line interface or code. It can then discover compatible devices, sequence work across them, monitor their outputs and alter parameters as results arrive. For high-speed or long-running work, the agent can package proven command sequences into deterministic code instead of reasoning over every action.

That last part is important. The goal is not to put a language model in the loop forever. It is to use one to explore and coordinate, then turn reliable procedures into inspectable automation.

## Early evidence, and the caveat

Anthropic's preview partners report examples across lab automation, robotics and quantum computing. In one quantum-laser experiment, Claude explored recovery strategies, then produced a deterministic relocking script that succeeded in 695 of 700 trials. In a Genentech lab-automation pilot, Claude coordinated a liquid handler, robotic arm and plate reader to tune pipetting parameters.

Those are promising vendor and partner reports, not a reason to let an agent freestyle around expensive hardware. Anthropic is explicit that current models still lack reliable physical intuition: in the Genentech tests, Claude needed human guidance to understand that bubbles were a physical failure rather than a software problem. It also sometimes paused for approval when an action looked risky. Good. Caution is a feature when the thing on the other end has motors, chemicals or lasers.

MHS is model-agnostic and intended to be open-sourced after the preview. For now, Anthropic is using the preview to develop safety evaluations and deployment practices with scientific labs, manufacturers and hardware vendors.

The connection to [Claude Fable 5.1](claude-fable-5-1.md) is obvious: a more capable long-horizon agent becomes more valuable once it can observe and coordinate real experiments. MHS is the plumbing that could make that possible without every lab reinventing it.

## Source

- [Previewing the Model Hardware Standard](https://www.anthropic.com/news/model-hardware-standard-research-preview)
