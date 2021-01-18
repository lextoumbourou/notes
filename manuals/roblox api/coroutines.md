# coroutine

Used to perform multiple tasks at once from the same script.

Coroutines `yield` when they want to give back control to the caller.

## Using Coroutines

Create a new coroutine by providing a function to `coroutine.create` like this:

```
local function task(...)
    coroutine.yield("return first")
    
    return "return second"
end

local coTest = coroutine.create(task)
```

You can then call `coroutine.resume(coTest, ...)`  to run the function up to the first yield.

```
local success, result = coroutine.resume(coTest, ...) -- prints "return first"
````

You can call `coroutine.status()` to inspect status of routine, which returns a string that represents the state of the coroutine:

* suspended: waiting to be resumed
* running: it's running now
* normal: awaiting yield of another routine (?)
* dead: returned or errored - can't be used further

You can use `coroutine.wrap` as a conveniemnt method to allow you to call the function continuously.