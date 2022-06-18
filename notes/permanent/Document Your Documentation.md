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

Documentation in multiple places isn't necessarily a problem -  information lends itself to different mediums.

However, making decisions about where to store a document every time we try to share information is a waste of value brain cycles and increases [Decision Fatigue](https://en.wikipedia.org/wiki/Decision_fatigue). We should instead direct that brain energy towards effective communication.

The solution is to make decisions once for each type of possible document you can think of and explain how and where they should be stored. It doesn't matter if you don't cover all kinds of information initially or if the system isn't perfect; we can improve it over time.

You can think of it as a code-style guide for your information.

It's also helpful to include *why* you have made that decision; that allows you or the future team to evaluate if the original logic still holds up. Maybe there's a better way to communicate that information now - that's okay.

A well-understood documentation system will help us to retrieve documents.

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

The cover is by [Jon Tyson on Unsplash](https://unsplash.com/@jontyson)