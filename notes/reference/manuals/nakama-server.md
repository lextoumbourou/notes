## Features

### User Accounts

* User is identity within a server
* Each user registered and has profile for other users to find, become friends or join groups and chat.
* User can own recors, share info with users and auth with various providers

#### Feature account

## Server run time

* The runtime framework is essential to writ server-side game logic for your game or app.
* You can write code you don't want to run on clients
* It can run authoritative logic or perform validation checks

### Load modules

* By defualt, server scans all files in data/modules.
    * Files with `.lua` or `.so` or `.js` extensions will be loaded and evaluated in startup sequence
    * Languages take precendence order: Go, Lua, JavaScript

### Examples

### RPC example


* Creating a Go script to add a `my_unique_id` rpc endpoint

```
import (
    "context",
    "database/sql",
    "encoding/json",
    "github.com/heroiclabs/nakama-common/runtime"
)

func InitModule(ctx context.Context, logger runtime.Logger, db *sql.DB, nk runtime.NakamaModule, initializer runtime.Initializer) error {
    if err := initializer.RegisterRpc("my_unique_id", SomeExample); err != nil {
        logger.Error("Unable to register:  %v", err)
        return err
    }
    
    return nil
}

func SomeExample(ctx context.Context, logger runtime.Logger, db *sql.DB, nk runtime.NakamaModule, payload string) (string, error) {

   meta := make(map[string]interface{})
   if err := json.Unmarshal([]byte(payload, ^meta); err != nil {
       return "", err
   }
   
   id := "SomeId"
   autoritative := false
   sort := "desc"
   operator := "best"
   reset := "0 0 * * 1"
   if err := nk.LeaderboardCreate(ctx, id, authoriative, sort, oeprator, reset, meta); err != nil {
       return "", err
   }
   
   return "Success", nil
}
```

### Register hooks

* For  all runtimes, the code will be evaluated as server startup. Can register funcitons which can operatoe on messages from clients to executre logic on demand.
    * Function called hooks

* All registered function receive `context` as first argument.
    * Can get info about the requests or user making it from the context

```
userId, ok := ctx.Value(runtime.RUNTIME_CTX_USER_ID).(string)
if !of {
// UserID Not in Context}
```

## Authoritative Multiplayer

* Nakama supports [[Client-Authoritative Multiplayer]] (aka relayed multiplayer) as well as [[Server-Authoritative Multiplayer]]
    * In relayed multiplayer, messages for clients are passed by server with inspection
        * It's up to the client to reconcile state changes between peers and perform arbitration on ambiguous or malcious messages from bad clients
        * Not useful if depending on central state managed by game server
    * In server authoritative, Nakama has a way to run custom match logic with fixed tick rate.
        * Messages can be validated and state chnges broadcast to connected peers
    * Can build:
        * Async real-time authoirtiatve multiplayer
            * Message sent to server, server calculates changes to envirronment nad players, and data is broadcast to relevant peers
                * Requires a high tick-rate for gameplay to feel responsive
        * Active turn-based multiplayer
            * Clash Royale style: Players expected to repsonst ot turns immediately.
                * Serer receives inputs, validates and broadcast to players
                * Tick-rate is low as rate of message send and recieved is low
        * Passive turn-based multiplayer
            * Words with Friends mobile examlpe
            * Gameplay can span several hours to weeks.
            * Server recieved input, validates them, stores in DB and broadcast changes to connected peers before shutting down
        * To support, introduce several concepts
    * Concepts
        * Match handler
            * Represents all server-side functions grouped together to handle game inputs and operate on them
            * 6 functions are required to process logic for a match a fixed rate on server
            * Server can run thousands of matches per machine
            * Match handler has an API to broadcast messages to connected players
        * Tick rate
            * Server periodically calls the match loop function, even when no input waiting to be processed
            * Can validate incoming input nad kick players who've been inactive
            * Periodic call is known as Tick Rate and respresents a desired fixed frequnecy which match should update
            * All incoming client data messages are queued until each tick when handed off to the match loop to be processed
            * Tick rate is a number of exectuions per second: 10 == once every 100ms.
        * Match state
            * A region of memory Nakama exposes to use for duration of match
            * Match handler governing each match may use this state to store any data it requires and given opportunity to update during tick
            * State can be thought of as result of continuous transformations applied to an inital state, based on loop of user input
        * Host node
            * Responsible for maintaining in-memory match state and allocating CPU resource to execute the loop at tick rate
                * Incomidng user input messages are waiting for the next tick to be processed are buffered to host node to ensure it is immediately avaialble
            * A single node is responsible for this to ensure highest level of conssitency accessing and updating the state, to avoid potential delay reconciling distributed state.
            * Match presents will still be replicates to all nodes in cluster have immediate access

### Create authoritative matches

* Manually
    * Use an RPC function which submits some user ID s to the server and cerate match
    * Mtch id will be created which could be sent out to players with in-app notification or push msesage 9or both)
* Use Matchmaker to find opponents and matchmaker matched callback on server to create an autoritative match and return a mathc ID.
    * Uses standard matchmaker API on client
* Client recieved matchmaker callback as normal with a mathc ID

### Join a match

* Player are not in match until they join even after matched by mathmaket
* Enables players to opt out of matches they decide not to play

### End authoritative matches

* Unlike relayed matches, authoritative multiplayer matches don't end when all players leave
* Normal and intended to allow you to suport use cases where players are allowed to temporarily disconnect while game workd continues to advance.
* They stop when callbacks reutnr a nil state.
    * Can do that at any point in the match

### Match listings

* can list matches that are active on server. Can filter based on exact-matches querlies on label field.
* If match was created with a label field of `skill=100-150`, you can filter down relevant matches

## Go runtime

### Match handler API

* The match handler that governs Authoritative Multiplayer is interface that uch implement all functions below.

```
MatchInit(ctx, logger, db, nk, params) -> (state, tickrate, label)
```

Invoked when a match is create as result of match create function and setup ip initial stat eof match. Called once at match state.

Parameters

* `ctx` - Context object represents information about match and server for information puposes
* `logger` - allows access to log messages at variable severity
* `db` - Database object that access underlying game database
* `nk` - `NakamaModule` exposes runtime functions to interact with various server systems and features
* `params` - map of params sent from `MatchCreate()` function while creating the match.
    * Could be list of matched users, their properties or any other relvant info to pas to the match

Returns

1. `state` (`interface{}`) - initial in-memory state of match.
    * May be any `interface{}` value that stores match state as progresse
2. `tickrate` - Repesented desired number of `MatchLoop()` calls per second. Between 1 and 60.
3. `label` `(string)`  - A string label for filter matches in listing operations. Must be between 0 and 2048 chars long.

#### MatchJoinAttempt

* Executed when user attempts to join match using client's match join operation
* Includes any rejoin request from client after lost connection is resumed

```MatchJoinAttempt(ctx, logger, db, nk, dispatcher, tick, state, presence, metadata) -> (state, result, reason)```

Returns:

* `state` (`interface{}`) - optionally updated state. May be non-nil value, or `nil` to end the match.
* `result` (`bool`) - `true` if join attempt is awlloed, false otherwise
* `reason` (`string`) - if join attempt should be rejected, optional string reject resaon can be returned from cletn

#### MatchJoin

* Run when one or more users have completed match join prcoess after their `MatchJoinAttempt()` returns `true`.
    * When presences are sent to this function, the users are ready to receive match data messages and can be targets for dispatcher's `BroadcastMessage()` function

Returns:

* `state` (`interface{}`) - An (optionally) updated state. May be non-nil value of `nil` to end match

#### MatchLeave

* Executed when one or more users have left match for any reason including connection loss.

#### MatchLoop

* Executed on interval based on tick rate returned by `MatchInit()` 