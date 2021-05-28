# Unit Testing Hubot Scripts

Hubot is pretty damn awesome, if you've found this article from Google, you probably know or at least have heard that by now. However, writing and testing Hubot scripts can be a bit of a pain, especially if you aren't that familiar with Node's ecosystem. 

This guide attempts to provide a simple-as-possible overview of testing the various types of Hubot scripts because I know how frustrating it can be. 

First things first, since this blog post is about testing, let me write the spec for this blog.

## The Requirements

* Mock all the things: Since we are unit testing no external dependancies shall be harmed in the making of these tests.
* Coverage measures.
* Automated lint checks.

## The Setup

Firstly, each Hubot script will be self-contained in a directory. In practise, however, I'd probably make a separate repository for each, as per the new [hubot-scripts]().

I'm going to add a couple of dependancies that will be consistent across each plugin. The first is ```hubot-mock-adapter```, basically a very simple Adapter (like Campfire, Slack or Flowdock adapters) for Hubot that emits events when certain common Hubot methods are called. I'm also going to add the ```node-jasmine``` BDD test framework, because I'm familiar with it, but you're free to use whichever tickles your fancy.

```
> mkdir tested-hubot-scripts && cd tested-hubot-scripts
> npm install node-jasmine hubot-mock-adapter --save
> mkdir test
```

I'm going to assume for the purpose of this article that you've already got a Hubot instance setup. If not, you should go aand follow the instructions [here](https://github.com/github/hubot/tree/master/docs)

Also, I'm going to setup a bit of boilerplate code in each repository that looks something like this:

```
> vi test/everyscript.spec.coffee
describe 'Every script', ->
  beforeEach -> 
    ready = false
    runs ->
      robot = new Robot(null, 'mock-adapter', false', 'testbot')
      
      robot.adapter.on 'connected', ->
        (require '../scripts/everyscript.coffee')(robot)

        user = robot.brain.userForId('1', user: 'someone', room: '#any')
        adapter = robot.adapter
        ready = true

    robot.run()

    waitsFor -> ready

  afterEach ->
    robot.shutdown()
```

I'll walk through it quickly:

* Lines 5: Create a new instance of Robot, passing in the 'mock-adapter.
* Line 7-8: We'll load the Hubot script on the ```on``` event, then create a reference the adapter object.
* Lines 12: The waitsFor method is a feature of Jasmine that will ensure the robot is fully loaded before running the tests

Okay, on to our tests...

## 1. The 80% case - testing a simple "hear and say" script

Let's say you want Hubot to listen for the word "hack" and return a random phrase from Hackers. You might write a plugin that'd look like [this](hack-the-planet.md).

You'll notice in it that there are a couple of complications to testing this. Firstly, we're using the ```msg.random``` method, to print out a random phrase. So, we should mock that out to ensure we know exactly what it's going to return.

With the spec setup as per section #0, we could add a new test that looks something like this:

```
  it "returns an appropriate phrase on 'hack'" (done) ->
    # Firstly, let's set the return value of msg.random
    robot.msg.random = () ->
      0

    adapter.on 'send', (envelope, strings) ->
      expect(strings[0]).toBe('Hack The Planet')
      done()

    adapter.receive new TextMessage user, 'hack'
```

Any questions?

Let's run it and watch the magic happen:

```
> npm test
```
