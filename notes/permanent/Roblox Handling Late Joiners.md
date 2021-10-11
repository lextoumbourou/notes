---

---

If your game has a round-based component, consider that a player may have entered the game at the start of the round, but may also be joining midday through the round.

One approach it to combine RemoteEvents and StringValues to get the best of both worlds.

Consider a game of tag. Each time a player becomes "it", we notify all the other players who is it. That everyone knows who to run away from.

Now, if a player joins late in the round, how do we tell them who's it?

We could either send an event when they join:

```
Player.Added:Connect(function()
    PlayerTaggedIt:FireClient()
end)
```

But there's a risk! What if the client isn't ready yet? What if the module that handlers `PlayerTaggerIt` hasn't fully loaded?

This is where the Roblox value series of properties can be handy.

When we fire the event, we can also define a `NumberValue` which defines who the user id of the current "it" person is.

So on the server we might write:

```
```