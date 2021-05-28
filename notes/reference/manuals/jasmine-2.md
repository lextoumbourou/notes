# Jasmine (2.0)

## ```describe```

* global Jasmine function
* takes two params
  * string that defines the spec
  * function that implements the spec
* used to group specs (in BDD style)
* Can nest ```describe``` blocks (top-level beforeEach is called first, then next and so on)
* Disable suits with ```xdescribe```

## ```it```

* function defined inside describe that actually defines the spec
* disable specs with ```xit```

## ```expect```

* Takes a value, called an 'actual'
* Takes a chain of matchers
* Examples:
  * ```expect(12).toBe(12);``` -- ```===``` style comparison
  * ```expect({'name': 'Bob'}).toEqual({'name': 'Bob'});``` -- use on object and arrays
  * ```expect('something').toMatch(/some/);``` -- regex matcher
  * ```expect(undefined).not.toBeDefined();``` -- test for undefined
  * ```expect(null).toBeNull();``` -- test for null
  * ```expect('something').toBeTruthy();``` -- test for truthy-ness
  * ```expect('').toBeFalsy();``` -- test for falsy-ness
  * ```expect([1, 2, 3]).toContain(1);``` -- match something in an array
  * ```expect(500).toBeLessThan(501);``` -- mathematical comparison
  * ```expect(500).toBeGreaterThan(499);``` -- mathematical comparison
  * ```expect(badFunction()).toThrow();``` -- ensure function raises an exception

## ```beforeEach()``` & ```afterEach()``` 

* called before spec
* use ```this``` to share objects between ```beforeEach``` and ```afterEach```.

## ```spyOn```

* ```expect(obj.someMethod).toHaveBeenCalled();```
* ```expect(obj.someMethod).toHaveBeenCalledWith(123);```
* To actually call the implementation, use ```and.callThrough()```
  * ```spyOn(obj, 'someMethod').and.callThrough();```
* To force a method to return a value use ```and.returnValue(someVal)```
  * ```spyOn(thing, 'random').and.returnValue(0);```
* To call another function when a spied on method is called, use ```and.callFake```
  * ```
    spyOn(thing, 'random').and.callFake(function() {
          return 1
     });
    ```
* ```calls.any()``` - returns ```false``` if spy has not been called, and ```true``` if called at least once
* ```calls.count()``` - returns number of times spy was called
* Heaps more

## ```jasmine.clock```

* Mock JS Timeout Function
* Install it with ```jasmine.clock().install```

## Async support

* pass ```1``` as second argument to ```beforeEach```, ```it``` and ```afterEach``` to put in async mode.
   * Spec will not start until ```done``` is called in ```beforeEach```
   * Spec will not complete until ```done``` is called again in ```it```
