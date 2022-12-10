# Roblox Datastore Considerations

The Roblox datastore is a pretty amazing service, though limited.

You can create a new Datastore just by creating an instance of the Datastore class.

## Periodic data syncing

One thing you quickly learn in Roblox is that you have a finite number of requests to the datastore you can make, so it's common to develop a pattern where you read the player's data into memory when they arrive, then perform updates to the dataset then persist it when they leave the server.

You might write a simple module like this:

Libraries like ProfileService do this for you

You may also choose to persist it periodically in case something goes wrong with the server.

Future requests to the players data are performed in memory.

However, there's some things to be careful of with this pattern:

* You must take care to stop persisting the player's data when they leave the server.
* If an issue occurs persisting data, the player may not be made aware.

Theres tools that handle this for you, like ProfileService.

## Be careful of using SetAsync!

When you have data that you want to update, you should always reach for UpdateAsync. When you have data you want to check before updating, UpdateAsync is also a good idea.

## 