---
title: Document Your Documentation
date: 2022-06-12 00:00
tags:
  - TechLeadership
summary: "A document that explains how your team's documentation works."
cover: /_media/document-documentation-cover.png
---

One simple thing that has helped every team I've been on: create a document that explains how your team's documentation works.

Over time, as your product's complexity increases, so does your documentation's complexity. It's common to find information scattered in many places.

For example, my team has most documents in Notion, some in the README.md of our product's repositories, some as comments in code, some pinned to Slack channels, and some in Google Docs.

Having documentation in multiple places isn't necessarily a problem. However, having us use valuable brain cycles thinking about where a document should live and how to present it is a waste. We should instead direct that brain energy toward effective communication.

The solution is to write a document that describes the different types of possible documentation and how and where we should present it. You can think of it as a code-style guide for your information.

For each documentation decision, it's helpful to include *why* you have made that decision; that allows the future team to evaluate if the original logic that leads to that decision still holds up.

A well-understood documentation system also helps a lot with document retrieval.

Here's an example:

> ## Documentation Overview
>
> ### Primary
>
> Notion should be the first choice for any documentation.
> 
> If it's product-specific, please put it in the folder under **Products > Product Name**.
> 
> Otherwise, please review the existing categories in the sidebar for the most relevant ones. You may add a new one if none fit.
>
> This ensures that our document is searchable and mostly in one place.
> 
> The following exceptions apply:
> 
> ### Code base setup info
>
> Put repository setup information and the code itself in a [README.md](http://readme.md) file . You should add a page in Notion that links to the project. See **Project A** for an example.
> Using the README for repository setup is usually the shortest path to ensuring that the setup instructions remain up-to-date and allow for linking to files within the repro.
>
> Use Notion if you need to expand on documentation beyond simply setting up the project.
>
> ### Ephemeral Documents
> 
> Some documents, for example, the project to-do lists, meeting summaries, balancing information, etc., make sense to live in a Google Document.
>
> Google Documents is generally better for these sorts of documents,  especially involving outside collaborators.
> 
> If the document needs to become long-lived, add a link for it in Notion. Each project has a page called **Key Documents** to which we can link these.
>

It doesn't matter if your document isn't the perfect approach right the first time. We can continually refine this based on what makes sense for our team.

The cover is by [Jon Tyson on Unsplash](https://unsplash.com/@jontyson)