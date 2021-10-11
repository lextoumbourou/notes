---
title: Roact notes
date: 2021-09-14 00:00
status: draft
---

Elements describe what you want your UI to look like at a single point in time. They're [immutable](https://en.wikipedia.org/wiki/Immutable_object): you can't change elements once they're created, but you can create new ones. Because creating elements is fast, this is no big deal.

Create new element using `Roact.createElement`

Pass Roblox class as first argument and properties as second argument.

Can pass children as 3rd argument of `createELement`


Notes from https://roblox.github.io/roact/guide/components/

Components are encapsulated, reusable pieces of UI that you can combine to build a complete UI.

Components accept inputs, kniown as props and return elements to describe the UI that should represent the inputs.

## Types of Components

Host Components

Function Components

Stateful Components

Roact also has _stateful_ components, which provide additional features like lifecycle methods and state. We'll talk about these features in a later section.

You can create a stateful component by calling `Roact.Component:extend` and passing in the component's name.