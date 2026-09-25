---
title: MM-REACT Prompting ChatGPT for Multimodal Reasoning and Action
date: 2024-10-04 00:00
modified: 2026-09-25 17:19
status: draft
category: paper
tags:
- AgenticReasoning
- LargeLanguageModels
paper_title: "MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action"
paper_url: https://arxiv.org/abs/2303.11381
paper_authors: Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, Lijuan Wang
paper_year: 2023
---

The paper, first released in March 2023, proposes **MM-REACT**, a system that integrates ChatGPT with a collection of computer vision models to solve complicated visual understanding tasks.

Unlike existing vision-language models, which require joint fine-tuning, MM-REACT uses ChatGPT's reasoning abilities to select and invoke specific computer vision models, making it a flexible, training-free approach. As the name suggests, it builds on the [ReAct: Synergizing Reasoning and Acting in Language Models](../../permanent/react-synergizing-reasoning-and-acting-in-language-models.md) pattern of interleaving reasoning with actions.

The paper highlights the capabilities of MM-REACT in various scenarios, such as visual maths and text reasoning, multi-image understanding and video summarisation, demonstrating its potential to address complex visual intelligence problems.

![Title and authors of MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action.](../../_media/mm-react-prompting-chatgpt-for-multimodal-reasoning-and-action-title.png)

![Grid of MM-REACT examples across nine capabilities, including visual maths, explaining a meme, locating a frisbee, following a bread recipe, totalling receipts, reading a bar chart, recognising brands and celebrities, and breaking a video into steps.](../../_media/mm-react-prompting-chatgpt-for-multimodal-reasoning-and-action-fig-1.png)

![Flowchart of MM-REACT. ChatGPT responds to the user and, if it produces a thought and action request, the named vision expert (such as image captioning, OCR or Bing search) is run and its output is fed back as an observation. Otherwise it responds to the user.](../../_media/mm-react-prompting-chatgpt-for-multimodal-reasoning-and-action-fig-2.png)

![Example conversation where MM-REACT answers questions about a floor plan image, including the number of bedrooms, room dimensions, kitchen appliances and a final summary.](../../_media/mm-react-prompting-chatgpt-for-multimodal-reasoning-and-action-arch-example.png)

![Example conversation where the user uploads four receipts (a flight, an Uber ride, groceries and a restaurant) and MM-REACT answers how much was spent on groceries, dining out, travel and taxes.](../../_media/mm-react-prompting-chatgpt-for-multimodal-reasoning-and-action-receipts.png)

![Step-by-step MM-REACT execution on a photo of two basketball players: ChatGPT calls captioning and face detection, then celebrity recognition to identify Kobe Bryant and Paul Pierce, then Bing search to answer a follow-up question about championship rings.](../../_media/mm-react-prompting-chatgpt-for-multimodal-reasoning-and-action-fig-3.png)
![Caption for Figure 3 explaining that numbered circles show the order of model calls, and grey text shows the internal thoughts and expert outputs hidden from the user.](../../_media/mm-react-prompting-chatgpt-for-multimodal-reasoning-and-action-fig-3-footer.png)