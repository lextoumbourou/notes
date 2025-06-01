---
title: "Absurdly Good Doggo Consistency with Flux.1 Kontext"
date: 2025-06-01 00:00
modified: 2025-06-01 00:00
summary: "Experiments with multi-turn character consistent editing"
tags:
- ImageEditing
- ImageGeneration
cover: /_media/flux1-experiments/doggo-convertable.jpg
---

<style>
table tr, table td {
   border: none;
}
</style>

This new image editing model from Black Forest Labs called [FLUX.1: Kontext](flux1-kontext.md) is really good.

You can read my paper summary here: [FLUX.1 Kontext: Flow Matching for In‑Context Image Generation and Editing in Latent Space](../reference/papers/flux1-kontext-flow-matching-for-in-context-image-generation-and-editing-in-latent-space.md)

Although GPT Image is still one of the <a href="https://notesbylex.com/imagen-4-is-faster-but-gpt-is-still-the-goat">best image models out there</a>, it is pretty limited in its ability to edit: characters get lost, and there's usually unrelated changes returned in the images.

On the other hand, FLUX.1: Kontext, thanks to approach of [Flow Matching](flow-matching.md) in latent space, maintains a high-quality level of text-to-image quality but with an absurdly good ability to edit photos. One remarkable thing is that it can maintain character consistency through many edits (called *multi-turn editing*). Even without the context of the chain of images, or any sort of in-painting, I found that it was able to keep a source character consistent, even after many rounds of editing.

To demonstrate the character consistency, I found the most recently taken photo of my dog, Doggo.

Doggo recently had TPLO surgery on each of her legs in two separate staggered surgeries due to some painful arthritis. She is fully recovered now and doing much better.

However, for the first two weeks after surgery, our poor puppy had to wear a giant cone to prevent her chewing off her stitches, which she hated.

![Original: A picture of my dog after TPLO surgery](_media/flux1-experiments/doggo-cone-1.jpg)

Let's see if Flux.1: Kontext can remove the cone from poor Doggo's head.

> "remove the cone from my dog's head"

![Updated Original - a very convincing removal of my Dog's cone](_media/flux1-experiments/doggo-cone-2.jpg)

Much better.

That works so well. Of course, if you look close enough, you can see the artifact where the cone was, but that's my dog, alright.

Again, the amazing thing about the Kontext model is its ability to do multi-turn editing. Using the most recent output as input, let's see if we can make my poor Dog look like her usual happy self:

> "make my dog look happy"

![_media/flux1-experiments/doggo-cone-3.jpg](_media/flux1-experiments/doggo-cone-3.jpg)

Admittedly, I had to try this a few times before I got something that looked convincing like this. One of them made her head too big, and the other did some weird stuff with her ears.

Anyway, that's pretty happy! I don't think she's ever quite smiled like that, but it's close.

Now, to maximise her happiness, I move her to one of her favourite places in the world:

> "change the background to a sunny beach scene."

![Doggo on the beach](../_media/flux1-experiments/doggo-cone-4-beach.jpg)

I'm really impressed by the cast shadows. The model has figure out where it wants the sun to be, and can generate shadows at roughly match that model. Wild.

Finally, to achieve peak happiness, I put her favourite chew toy next to her, a red deer antler.

> "add an antler bone in front of her"

![Doggo with antler bone in front of her](../_media/flux1-experiments/doggo-cone-5-beach.jpg)

Even after four rounds of editing, it still looks exactly like Doggo - really impressive stuff.

She looks so happy; I think this could be a birthday card.

My nephew's birthday is coming up, and he loves Minecraft. So I'll try turning it into a Minecraft-themed bday.

> Change background to minecraft. Write "Happy Birthday, Nephew" in bright, colorful text on top of the image.

![doggo-cone-6.jpg](../_media/flux1-experiments/doggo-cone-6-bday.jpg)

Looking pretty good, albeit a little distorted.

I wonder if I can use the same photo for all my future greeting card needs. It's June, but still, it's never too early to be planning Christmas. It is coming into winter in the Southern Hemisphere, after all. Isn't Christmas in July a thing?

> change text to "Seasons Greetings" with a Christmas font. Convert into a snowy background. Remove antler. Add a snowman next to her. Add a Christmas hat on top.

![doggo-xmas-1.jpg](../_media/flux1-experiments/doggo-xmas-1.jpg)

Amazing. Now we're seven edits deep, and it still looks like my Doggo. Granted, we've started to see some artifacts, and there's some roughness around the edges, but this is looking good.

Flux.1 also excels at style transfer, so let's try a few different styles for the Christmas Card.

<table style="width:100%; table-layout: fixed;">
  <tr>
    <td style="text-align:center; vertical-align:top;">
      <img src="../_media/flux1-experiments/doggo-90s-christmas.jpg" width="150" /><br>
      <span style="font-size:smaller;">"Convert into the style of a 90s Christmas Movie poster"</span>
    </td>
    <td style="text-align:center; vertical-align:top;">
      <img src="../_media/flux1-experiments/doggo-vintage.jpg" width="150" /><br>
      <span style="font-size:smaller;">"Convert into a Vintage Storybook Style Christmas Card."</span>
    </td>
    <td style="text-align:center; vertical-align:top;">
      <img src="../_media/flux1-experiments/doggo-illustration.jpg" width="150" /><br>
      <span style="font-size:smaller;">"Convert into a watercolor illustrated Christmas Card."</span>
    </td>
    <td style="text-align:center; vertical-align:top;">
      <img src="../_media/flux1-experiments/doggo-simpsons.jpg" width="150" /><br>
      <span style="font-size:smaller;">"change into the style of The Simpsons"</span>
    </td>
  </tr>
</table>

Few interesting variations on the text, but very impressive nonetheless.

Back to the original Christmas Card, the Kontext paper demonstrates even more incredible global edits, like adding multiple characters, and rotating camera angles.

![figure1.png](../_media/flux1-experiments/figure1.png)

*Figure 1: Consistent character synthesis with FLUX.1 Kontext by Black Forest Labs*

Let's experiment with some of that.

> Remove text. There are now two dogs driving in a pink convertible.

![doggo-convertable.jpg](../_media/flux1-experiments/doggo-convertable.jpg)

I mean, that kind of works. I think it's still my dog in the driver's seat. Hard to tell whether the snowman or Doggo is driving the car, but between them I assume they've got it covered.

I tried the prompt "Watch them from behind.", which is actually given as an example in the paper. That was immediately flagged as NSFW and refused. Flux, you definitely are misunderstanding me.

I tried an alternate prompt.

> "turn the camera to watch them from the back"

![doggo-car.jpg](../_media/flux1-experiments/doggo-car.jpg)

Not sure exactly what's going on here, but it has turned the car around, but not any of the character. Starting to get a little terrifying, feeling a bit like [Loab](https://en.wikipedia.org/wiki/Loab) might be waiting a few turns down the line, so I'll stop.

Now, the showcase of all the edits:

## Flux.1: Kontext - Character Consistency

<table style="width:100%; table-layout: fixed;">
  <tr>
    <td style="text-align:center; vertical-align:top;">
      <img src="_media/flux1-experiments/doggo-cone-1.jpg" width="150" /><br>
      <span style="font-size:smaller;">Source image</span>
    </td>
    <td style="text-align:center; vertical-align:top;">
      <img src="_media/flux1-experiments/doggo-cone-2.jpg" width="150" /><br>
      <span style="font-size:smaller;">"remove the cone from my dog's head"</span>
    </td>
    <td style="text-align:center; vertical-align:top;">
      <img src="_media/flux1-experiments/doggo-cone-3.jpg" width="150" /><br>
      <span style="font-size:smaller;">"make my dog look happy"</span>
    </td>
    <td style="text-align:center; vertical-align:top;">
      <img src="_media/flux1-experiments/doggo-cone-4-beach.jpg" width="150" /><br>
      <span style="font-size:smaller;">"change the background to a sunny beach scene"</span>
    </td>
  </tr>
  <tr>
    <td style="text-align:center; vertical-align:top;">
      <img src="_media/flux1-experiments/doggo-cone-5-beach.jpg" width="150" /><br>
      <span style="font-size:smaller;">"add an antler bone in front of her"</span>
    </td>
    <td style="text-align:center; vertical-align:top;">
      <img src="_media/flux1-experiments/doggo-cone-6-bday.jpg" width="150" /><br>
      <span style="font-size:smaller;">"Change background to Minecraft. Write 'Happy Birthday, Nephew' in bright, colorful text"</span>
    </td>
    <td style="text-align:center; vertical-align:top;">
      <img src="_media/flux1-experiments/doggo-xmas-1.jpg" width="150" /><br>
      <span style="font-size:smaller;">"Seasons Greetings, snowy background, snowman, Christmas hat, remove antler"</span>
    </td>
    <td style="text-align:center; vertical-align:top;">
      <img src="_media/flux1-experiments/doggo-convertable.jpg" width="150" /><br>
      <span style="font-size:smaller;">"Two dogs driving in a pink convertible"</span>
    </td>
  </tr>
</table>

It's a very impressive model indeed. We can contrast the same sequence of turns with gpt-image-1, which the paper reports as the second best performing model for character consistency.

![character-ref.png](../_media/flux1-experiments/character-ref.png)

*Figure 9: Image-to-image evaluation on KontextBench by Black Forest Labs*

## gpt-image-1 - Character Consistency ref

<table style="width:100%; table-layout: fixed;">
  <tr>
    <td style="text-align:center; vertical-align:top">
      <a href="_media/flux1-experiments/doggo-cone-1.jpg" target="_blank">
        <img src="_media/flux1-experiments/doggo-cone-1.jpg" width="150" />
      </a><br>
      <span style="font-size:smaller;">Source image</span>
    </td>
    <td style="text-align:center; vertical-align:top;">
      <a href="_media/flux1-experiments/doggo-gpt-image-1.png" target="_blank">
        <img src="_media/flux1-experiments/doggo-gpt-image-1.png" width="150" />
      </a><br>
      <span style="font-size:smaller;">1. "remove the cone from my dog's head" (gpt-image-1)</span>
    </td>
    <td style="text-align:center; vertical-align:top;">
      <a href="_media/flux1-experiments/doggo-gpt-image-2.png" target="_blank">
        <img src="_media/flux1-experiments/doggo-gpt-image-2.png" width="150" />
      </a><br>
      <span style="font-size:smaller;">2. "make my dog look happy" (gpt-image-1)</span>
    </td>
    <td style="text-align:center; vertical-align:top;">
      <a href="_media/flux1-experiments/doggo-gpt-image-3.png" target="_blank">
        <img src="_media/flux1-experiments/doggo-gpt-image-3.png" width="150" />
      </a><br>
      <span style="font-size:smaller;">3. "change the background to a sunny beach scene"</span>
    </td>
  </tr>
  <tr>
    <td style="text-align:center; vertical-align:top;">
      <a href="_media/flux1-experiments/doggo-gpt-image-4.png" target="_blank">
        <img src="_media/flux1-experiments/doggo-gpt-image-4.png" width="150" />
      </a><br>
      <span style="font-size:smaller;">4. "add an antler bone in front of her"</span>
    </td>
    <td style="text-align:center; vertical-align:top;">
      <a href="_media/flux1-experiments/doggo-gpt-image-5.png" target="_blank">
        <img src="_media/flux1-experiments/doggo-gpt-image-5.png" width="150" />
      </a><br>
      <span style="font-size:smaller;">5. "Change background to minecraft. Write 'Happy Birthday, Nephew' in bright, colorful text on top of the image."</span>
    </td>
    <td style="text-align:center; vertical-align:top;">
      <a href="_media/flux1-experiments/doggo-gpt-image-6.png" target="_blank">
        <img src="_media/flux1-experiments/doggo-gpt-image-6.png" width="150" />
      </a><br>
      <span style="font-size:smaller;">6. "change text to 'Seasons Greetings'. Convert into a snowy background. Remove antler. Add a snowman next to her. Add a Christmas hat on top."</span>
    </td>
    <td style="text-align:center; vertical-align:top;">
      <a href="_media/flux1-experiments/doggo-gpt-image-7.png" target="_blank">
        <img src="_media/flux1-experiments/doggo-gpt-image-7.png" width="150" />
      </a><br>
      <span style="font-size:smaller;">7. "Remove text. There are now two dogs driving in a pink convertible."</span>
    </td>
  </tr>
</table>

Still an incredible model, but we can see clearly that even by the second image it's a totally different dog.

Black Forest Labs cooked with this one, as the kids would say.