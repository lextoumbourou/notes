---
title: Flux
date: 2025-12-19 00:00
modified: 2025-12-19 00:00
status: draft
---

Flux is one of the two [Reactive APIs](../../../permanent/reactive-apis.md) provided as part of Java's [Project Reactor](../../../permanent/project-reactor.md). It represents a **reactive stream** that can emit 0 to N elements, and then complete (either successfully or with an error). Unlike a Mono object, which emits 0 or 1 element (used for finding a single element), or just doing a task where you don't care about the output.

It's a similar idea to a [Python Generator](python-generator.md) or [Lua Coroutines](lua-coroutines.md) in the sense it models a sequence of values over time, but Flux is push-based, asynchronous and reactive, with the producer controlling when values are emitted.

This article walks through a demonstration of Flux, from absolute scratch, assuming basically no Java knowledge.

Let's firstly initialise a project using Maven that we can use to test Flux.


```
mvn archetype:generate -DgroupId=com.example.reactive -DartifactId=reactive-quotes -DarchetypeArtifactId=maven-archetype-quickstart -DarchetypeVersion=1.4 -DinteractiveMode=false
```

Then, I'm going to use Java version 17, and add **Project Reactor** to `pom.xml`

```
<properties>
    <maven.compiler.source>17</maven.compiler.source>
    <maven.compiler.target>17</maven.compiler.target>
</properties>
```

```
<dependencies>
    <dependency>
        <groupId>junit</groupId>
        <artifactId>junit</artifactId>
        <version>4.11</version>
        <scope>test</scope>
    </dependency>

    <!-- Project Reactor -->
    <dependency>
        <groupId>io.projectreactor</groupId>
        <artifactId>reactor-core</artifactId>
        <version>3.6.8</version> <!-- or latest -->
    </dependency>
</dependencies>
```

The first thing I'm going to do, is use the `just` method, to output a list of Karl Pilkington quotes. Then creates a new Flux object, whose only job is to emit some provided events and complete.

In 
``` java
package com.example.reactive;
import reactor.core.publisher.Flux;

/**
 * Hello world!
 *
 */
public class App
{
    public static void main( String[] args )
    {
        System.out.println( "Starting KP Quotes App..." );

        Flux<String> quotes = Flux.just(
                "You never see an old man eating a Twix.",
                "I find that if you just talk, your mouth comes up with stuff.",
                "The cafe was called Tattoos. The fella who owned it didn't have any tattoos... but we never saw his wife.",
                "Stay green, stay in the woods, and stay safe.",
                "Stop looking at the walls, look out the window."
        );
    }
}
```

Then run it like so:

```
mvn clean package
mvn exec:java -Dexec.mainClass="com.example.reactive.App"
```

There's no output yet. Why's that? Because we don't have anything subscribing the the Flux app yet. A Flux object has a method call `subscribe` that allows you to consume each element that the Flux emits.


```java
subscribe(@Nullable java.util.function.Consumer<? super T> consumer)
```

I'll add that to the `App`:

```java
// ...
Flux<String> quotes = Flux.just(
 // ...
);

quotes.subscribe(quote -> System.out.println("Quote: " + quote));
```

You should now see the quotes logged to stdout:

```
[INFO] --- exec:3.6.2:java (default-cli) @ reactive-quotes ---
Starting KP Quotes App...
Quote: You never see an old man eating a Twix.
Quote: I find that if you just talk, your mouth comes up with stuff.
Quote: The cafe was called Tattoos. The fella who owned it didn't have any tattoos... but we never saw his wife.
Quote: Stay green, stay in the woods, and stay safe.
Quote: Stop looking at the walls, look out the window.
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
```

The subscribe method also accepts an errorConsumer and a completeConsumer, which runs once all items have been emitted.

```java
quotes.subscribe(
        quote -> System.out.println("Quote: " + quote),
        error -> System.err.println("Error: " + error),
        () -> System.out.println("All quotes emitted!")
);
```

You won't see the error, but you should see the final line:

```
All quote emitted!
```

### Challenge 1

Can you turn that into a list of integers, returning a `Flux<Integer>` and print the squared value for each integer?

---

We can also transform streams with typical functional program converts like `map` and `filters`. For example, we can convert all the quotes to uppercase before subscribing

```java
// ...
Flux<String> quotes = Flux.just(
 // ...
);
Flux<String> uppercased = quotes.map(String::toUpperCase);

uppercased.subscribe(quote -> System.out.println("Quote: " + quote));
```

```
Starting KP Quotes App...
Quote: YOU NEVER SEE AN OLD MAN EATING A TWIX.
Quote: I FIND THAT IF YOU JUST TALK, YOUR MOUTH COMES UP WITH STUFF.
Quote: THE CAFE WAS CALLED TATTOOS. THE FELLA WHO OWNED IT DIDN'T HAVE ANY TATTOOS... BUT WE NEVER SAW HIS WIFE.
Quote: STAY GREEN, STAY IN THE WOODS, AND STAY SAFE.
Quote: STOP LOOKING AT THE WALLS, LOOK OUT THE WINDOW.
All quote emitted!
```
or filter out any short strings that are 100 chars or longer:

```java
// ...
Flux<String> quotes = Flux.just(
 // ...
);
Flux<String> shortQuotes = quotes.filter(q -> q.length() > 100);

shortQuotes.subscribe(quote -> System.out.println("Quote: " + quote));
```

## Challenge 2

* Filter quotes that contain the letter "w"
* Map them to include their length "Quote (len=42): ...".

---

