# Chapter 3: Programming with RDDs

* All work is either:
  * Creating new RDDs
  * Transforming existing RDDs
  * Or calling operations on RDDs to compute a result

## RDD Basics

* An immutable distributed collection of objects

* Can be created in two ways:
  * by loading an external dataset (like ``sc.textFile('README.md')``)
  * or by distributing a collection of objects (like a list or set) into their driver program

* Once created, offer two types of operations: *transformations* and *actions*
  * Transform example: ``lines = lines.filter(lambda line: 'Some string' in line)``
  * Action example: ``lines.count()``
  * Spark computes RDDs lazily: only once first action is requested. Transforms are not run until *actions* are requested. 

* Computed each time actions are run on them. To persist, use the ``persist`` method.

```
> lines.persist(StorageLevel.MEMORY_ONLY_SER)
```

## Creating RDDs 

* To load an in-memory object, use the ``parallelize(someobj)`` method:

```
In [1]: peoplez = ['john', 'sally', 'bobbo', 'barry']

In [2]: lines = sc.parallelize(peoplez)

In [3]: lines.count()
Out[3]: 4
```

* Can also be loaded from external sources, as per earlier examples.

## RDD Operations

* Transforms return a new RDD, does not mutate an existing RDD (they're immutable).
* Can perform a union across two RDDs, as follows:

```
errors = input.filter(lambda x: 'error' in x)
warnings = input.filter(lambda x: 'warning' in x)
badlines = errors.union(warnings)
```

* Take method is an action akin to ``head`` in bash.

``
badlines.take(10)
``

* Lazy Evaluation: Don't think of an RDD containing specific data, but a set of instructions on how to compute data.

## Passing Functions to Spark

* Avoid passing references to objects in ``actions`` as Spark tries to serialize the whole thing.

```
## BAD!
class SomeComputation(object)
    # ..
    def get_matches(self, rrd):
      return rdd.filter(self.isMatch)

## GOOD
class SomeComputation(object)
    # ..
    def get_matches(self, rrd):
      resut = self.isMatch
      return rdd.filter(result)
```

## Common Transformations and Actions

*Not started*
