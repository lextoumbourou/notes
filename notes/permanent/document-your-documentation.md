---
title: Document Your Documentation
date: 2022-06-18 00:00
tags:
  - SoftwareEngineering
  - KnowledgeManagement
summary: "A short document that explains how your team documents things."
cover: /_media/document-documentation-cover.png
---

One simple thing that can help a software team is to create a short document explaining how the team documents things.

Over time, as your product's complexity increases, so does the complexity of knowledge required to support it. It's common to find information scattered in many places.

My team has most documents in Notion, some in the README.md files, some as comments in code, some pinned to Slack channels, and some in Google Docs.

Documentation in multiple places isn't necessarily a problem - information requires various mediums.

However, having to make a decision every time we try to share information is a waste of valuable brain cycles and increases [Decision Fatigue](https://en.wikipedia.org/wiki/Decision_fatigue). I would prefer we utilize our brain energy for writing good documentation.

We should instead make a decision once for each class of information (see example below), documenting how and where we store and retrieve as clearly as possible. It doesn't matter if we don't cover all kinds of information initially or if the system isn't perfect; we can improve it over time.

You can think of it as a code-style guide for your information.

It's also helpful to include *why* you have made decisions; that allows you or the future team to evaluate if the original logic still holds up. Maybe there's a better way to communicate that information now - that's okay.

A well-understood knowledge system will also help us with document retrieval.

Here's an example:

> ## Documentation Guide
>
> ### Primary
>
> Notion should be the first choice for most documentation.
>
> If it's product-specific, please put it in the folder under **Products > Product Name**.
>
> Review the existing categories in the sidebar for the most relevant ones for company-wide sharing. You may add a new one if none fit.
>
>  Notion ensures our documentation is searchable and easily accessible.
>
> Some exceptions apply:
>
> ### Repository setup information
>
> A `README.md` file should describe how to set up the project, including running the application and unit tests.
> Add a page in Notion that links to each repository. See **Project A** for an example.
> Use Notion if you need to expand on documentation beyond simply setting up the project.
>
> Using the README for repository setup is usually the shortest path to ensuring that the setup instructions remain up-to-date and allow for linking to files within the repro. It is also the first place that most newer developers will look for documentation.
>
> ### Ephemeral Documents
>
> Some documents, for example, the project to-do lists, meeting summaries, balancing information, etc., make sense to live in a Google Document.
>
> Google Documents is generally better for these sorts of documents, especially involving outside collaborators.
>
> If the document needs to become long-lived, add a link for it in Notion. Each project has a page called **Key Documents** to which we can link these.
>

The cover is by [Jon Tyson on Unsplash](https://unsplash.com/@jontyson)
