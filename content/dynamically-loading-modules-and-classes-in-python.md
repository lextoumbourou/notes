Title: Dynamically loading modules and classes in Python
Tagline: Using pkgutil and getattr to do exactly as the title implies
Slug: dynamically-loading-modules-and-classes-in-python
Date: 2012-10-12

</p>

<div class="intro">
There was once a time, I remember, many, many minutes ago, where I
wished to implement a rudimentary plugin engine for a little
application. In it, there would be a plugin directory where Python
modules could be copied and then dynamically loaded on application
start.
</div>

### The setup

The expectations for plugins would be that the class's name matches its
file's name, PascalCased after removing the *plugin\_* prefix. For
example, for module *plugin\_do\_something.py*, the class name will be
*DoSomething()*. So, the setup should look something like this:

    :::bash
    lex@server:~/my_app> ls -l plugins/
    total 3
    -rw-r--r-- 1 lex lex   0 Oct 11 13:35 __init__.py
    -rw-r--r-- 1 lex lex  55 Oct 11 13:47 plugin_test.py
    -rw-r--r-- 1 lex lex  57 Oct 11 13:47 plugin_test_2.py
    lex@server:~/my_app> cat plugins/plugin_test.py
    class Test():
        def run(self):
            print "Inside the Test class!"
    lex@server:~/my_app> cat plugins/plugin_test_2.py
    class Test2():
        def run(self):
            print "Inside the Test2 class!"

### Iterating through modules for fun and profit

The [pkgutil][] package includes a function called *iter\_modules*,
which allows for iterating through a list of modules in a package
(usually, a directory with a *\_\_init\_\_.py* file in it.)

*iter\_modules* takes in a list of paths (and, additionally, a prefix to
append to each module) and yields a list of tuples containing a
[loader][], the module name and an *ispkg* bool. So, your code could
look something like this:

    :::python
    import pkgutil
    modules = pkgutil.iter_modules(path=["plugins"])
    for loader, mod_name, ispkg in modules: 
        print mod_name


Which should run something like this:

    :::bash
    lex@server:~/my_app> python dynamic_loader.py
    plugin_test
    plugin_test_2

### Determining the class name

Now that I have the module names, I can determine the class names. Since
we know the format the class should be in, we can get the class name
with a function like this:

    :::python
    def get_class_name(mod_name):
        """Return the class name from a plugin name"""
        output = ""

        # Split on the _ and ignore the 1st word plugin
        words = mod_name.split("_")[1]

        # Capitalise the first letter of each word and add to string
        for word in words:
            output += word.title()
        return output

### Importing the modules and instantiating the classes

Utilising the [\_\_import\_\_][] function, whose purpose is to "import a
module whose name is only known at runtime", we can dynamically load the
module. Then, we can call the [getattr][] function to create a reference
to a class, which we use to instantiate an instance. In other words,
this:

    :::python
    loaded_mod = __import__(path+"."+mod_name, fromlist=[mod_name])
    # Get the class's name
    class_name = get_class_name(mod_name)

    # Load it from imported module
    loaded_class = getattr(loaded_mod, class_name)

    # Create an instance of it
    instance = loaded_class()

### Putting it all together

Finally, we put it all together and call the *run()* method that I
specified in each class.

    :::python
    import pkgutil
    import sys
    import os

    def get_class_name(mod_name):
        """Return the class name from a plugin name"""
        output = ""

        # Split on the _ and ignore the 1st word plugin
        words = mod_name.split("_")[1:]

        # Capitalise the first letter of each word and add to string
        for word in words:
            output += word.title()
        return output

    path = os.path.join(os.path.dirname(__file__), "plugins")
    modules = pkgutil.iter_modules(path=[path])

    for loader, mod_name, ispkg in modules:
        # Ensure that module isn't already loaded
        if mod_name not in sys.modules:
            # Import module
            loaded_mod = __import__(path+"."+mod_name, fromlist=[mod_name])
           
            # Load class from imported module
            class_name = get_class_name(mod_name)
            loaded_class = getattr(loaded_mod, class_name)

            # Create an instance of the class
            instance = loaded_class()
            instance.run()

And, now to profit.

    :::bash
    lex@server:~/my_app> python dynamic_loader.py
    Inside the Test class!
    Inside the Test2 class!

### Following me on Twitter

While you're here, you might as well follow me on [Twitter][].

  [pkgutil]: http://docs.python.org/library/pkgutil.html
  [loader]: http://www.python.org/dev/peps/pep-0302/
  [\_\_import\_\_]: http://docs.python.org/library/functions.html#__import__
  [getattr]: http://docs.python.org/library/functions.html#getattr
  [Twitter]: http://twitter.com/lexandstuff
  [comments powered by Disqus.]: http://disqus.com/?ref_noscript
  [comments powered by <span class="logo-disqus">Disqus</span>]: http://disqus.com
