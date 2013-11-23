Title: Integrating Bootstrap with Django using Less and Fabric
Slug: integrating-bootstrap-with-django-using-less-and-fabric
Date: 2012-11-11

### Overview

</p>

It can be frustrating finding the perfect way to work with
[Bootstrap][]. On the one hand, I want to make sites look as customised
and non-Bootstrappy as possible. On the other hand, it'd be nice to have
an easy way to pull down the updates from the main Bootstrap repo
without going through merge hell, or - god forbid - having to copy and
paste bits and pieces by hand.

</p>

I spent a day doing research into the ways other people were managing
this, and have found that people are solving the problem a lot of
different [ways][]. None of which seemed satisfyingly "perfect". Though,
I couldn't quite put my finger on what perfect should be. So, I decided
I would set myself some simple guidelines for exactly what I wanted out
of my Bootstrap experience and work out a solution to each.

</p>

The strategy I've come up with is probably not 100% ideal but then
neither is your face (joking, you're beautiful). I also plan to update
this post if I come up with a better way of doing it. Feel free to let
me know in the comments if you've got a better way of doing it. Anyway,
on to...

</p>

### The Requirements

</p>

I want to be able to...

1.  Custom Bootstrap classes and style variables using Less.
    </p>
    <p>
2.  Easily pull updates from the main Bootstrap repo without overwriting
    any of my custom variables.
    </p>
    <p>
3.  Ensure my Less files are compiled server-side in production but
    client-side during development, so I can make changes and see
    results immediately.
    </p>
    <p>

</p>

* * * * *

</p>

### Requirement 1: Customise base Bootstrap classes and styles using Less

</p>

With a little help from my [friends][], I decided upon this solution. I
would take the `less` directory out of the Bootstrap repo and put it in
my static files directory. Then, I created my own `variables.less` file
which would start its life as a copy of the one provided by Bootstrap
but allow me to update and edit variables easily. Additionally, I have a
`custom.less` file that references the variables file and allows me to
overwrite Bootstrap classes and so forth. Let me break it down for
ya'll.

</p>

\1. Firstly, in my Django project, I've setup a `less` directory in the
`templates` directory of my project:

</p>

<p>
    lex@server:~/> cd project/templates/staticlex@server:~/project/templates/static/> mkdir less

</code>

</p>

\2. Then, I cloned the Bootstrap repo and copied the `less` directory
into my project's directory, renaming it to `bootstrap`. Like so:

</p>

<p>
    lex@server:~/project/templates/static/> cd ~/srclex@server:~/src> git clone git://github.com/twitter/bootstrap.gitlex@server:~/src> cp bootstrap/less ~/project/templates/static/less/bootstrap

</code>

</p>

\3. Next, at the root of my `less` directory, following the
[StackOverflow post][friends], I created a `custom.less` file which
references the to the Bootstrap modules in the copied subdirectory and
created the updated `variables.less` files as mentioned earlier. So, it
looks something like this:

</p>

##### project/templates/static/less/custom.less

</p>
<p>
    //...// CSS Reset@import "bootstrap/reset.less";// Core variables and mixins@import "variables.less"; // Modify this for custom colors, font-sizes, etc@import "bootstrap/mixins.less";// Grid system and page structure@import "bootstrap/scaffolding.less";@import "bootstrap/grid.less";@import "bootstrap/layouts.less";//...

</code>

</p>

And I added my CSS overrides in the `custom.less` file immediately after
the import statements.

</p>

##### project/templates/static/less/custom.less

</p>
<p>
    // Utility classes@import "bootstrap/utilities.less"; // Has to be last to override when necessary// Brand: website or project name// -------------------------.navbar .brand {    float: left;    display: block;       // Vertically center the text given @navbarHeight    padding: ((@navbarHeight - @baseLineHeight) / 2) 20px ((@navbarHeight - @baseLineHeight) / 2);    margin-left: -20px; // negative indent to left-align the text down the page    font-size: 40px;    font-weight: bold;    letter-spacing: -2px;    color: @navbarBrandColor;    &:hover {       color: darken(@navbarBrandColor, 20%);    }}// Allow us to put symbols and text within the input field for a cleaner look.input-append,.input-prepend {    margin-bottom: 5px;    font-size: 0; // white space collapse hack    white-space: nowrap; // Prevent span and input from separating    // Reset the white space collapse hack    input,    select,    .uneditable-input,    .dropdown-menu {        font-size: @baseFontSize;     }  input,  select,  .active {      border-color: #ccc;    }  }

</code>

</p>

\4. Finally, to ensure that all works, I can update my `base.html`
parent template and put a reference to my new less file. I should be
able to update less values and the page should be dynamically updated.
But first, I need to install the less js file which can be found
[here][].

</p>

<p>
    lex@server:~/> cd project/templates/static/js/lex@server:~/project/templates/static/js/> wget http://cloud.github.com/downloads/cloudhead/less.js/less-1.3.1.min.js

</p>

Now client side I can compile the less file for testing by adding the
following lines of html.

##### project/templates/base.html

</p>
<p>
    <link rel="stylesheet/less" type="text/css" media="all" href="{{STATIC_URL}}less/theme.less" /><script src="{{STATIC_URL}}js/less-1.3.1.min.js"></script>

</p>

And that works! So that's requirement 1 complete. On to the second.

</p>

### Requirement 2: Easily pull updates from the main Bootstrap repo without overwriting any of my custom variables

</p>

The simplest way to handle any maintenance task in Python projects, is
usually to add a function to a project's [Fabric][] script. Here, I
added a function called `update_bootstrap()` which takes an option [Git
tag][] argument, and pulls down the latest Bootstrap updates, checking
out to the tagged version, if specified, then copying the **less**
directory into my project as specified in Requirement 1.

</p>

If you don't have a `fabfile.py` or you are new to Fabric, you're
welcome to skip this step. However, if that's the case, then now is a
good time to get your learn on. Fabric is amazing.

</p>

##### project/fabfile

</p>
<p>
    import osfrom fabric.api import run, settings, env, cdfrom fabric.contrib import filesdef local():    env.hosts = ['localhost']def update_bootstrap(tag=None):    """    Update Bootstrap files to tag version. If a tag isn't specified just    get latest version.    """    repo = 'https://github.com/twitter/bootstrap'    local_path = os.path.join(os.path.dirname(__file__),                             'templates/static/less/bootstrap')    with settings(warn_only=True):        # Create the source directory if it doesn't exist        if not files.exists('~/src/'):            run('mkdir ~/src')        # Clone the Bootstrap project if we don't have a local copy of the repo        if not files.exists('~/src/bootstrap'):            with cd('cd ~/src'):                run('git clone {0}'.format(repo))        # Pull down updates        with cd('~/src/bootstrap'):            run('git pull origin master')            # Checkout to tag if specified            if tag:                run('git checkout {0}'.format(tag))            # Remove the project's Bootstrap files            run('rm -vR {0}'.format(local_path))            # Copy the updated files into the project            run('cp -vfR less {0}'.format(local_path))

</p>

\2. Then to run it...

</p>

<p>
    lex@server:~/> fab local update_bootstrap

</p>

### Requirement 3: Less files are compiled server-side in production but client-side during development

</p>

Nothing kills the fun of messing around with a page's design more than
having to compile CSS after every fiddle. There's a couple of choices
here. I could either run a script like [this][] that watches for updates
to less files and automatically compiles them, but it's a bit of a
hassle. Instead, I went with the idea proposed by [Julia Elman][], using
the Django Compressor precompiler.

</p>

[Django Compressor][] is a Django app that allows you to compress linked
and inline JavaScript or CSS into a single cached file. Additionally,
and most importantly in this context, it allows a person to specify
precompilers of their choice. In this example, I'm going to use the Less
compiler, `lessc`...

\1. Firstly, I ensured I have the Less compiler installed, which should
be done via **npm**. I included the step to install **npm** for
completeness.

</p>

<p>
    lex@server:~/> sudo apt-get install npmlex@server:~/> sudo npm install -g less

</p>

\2. Next, I installed django-compressor via Pip.

</p>

<p>
    lex@server:~/> sudo pip install django-compressor

</p>

\3. Then, I added django-compressor to my `INSTALLED_APPS` tuple in the
`settings.py` file.

<p>
    #...INSTALLED_APPS = (    #...    'compressor',    #...

</p>

\4. After, I specified Less as a precompiler by adding a
`COMPRESS_PRECOMPILERS` tuple in the same file.

</p>

<p>
    #...COMPRESS_PRECOMPILERS = (    ('text/less', 'lessc {infile} {outfile}'),)

</p>

\5. Since I am using Django’s staticfiles contrib app, I have to add
Django Compressor’s file finder to the `STATICFILES_FINDERS` tuple.

</p>

<p>
    STATICFILES_FINDERS = (    #...    'compressor.finders.CompressorFinder',    #...)

</p>

\6. Now, in my parent template, I configured it such that, if DEBUG was
enabled (which definitely should only be enabled in dev), then I would
compile the CSS client-side, however, in production we would use
`django-compressor`. Like so:

</p>

<p>
    {% load compress %}<!DOCTYPE html><html lang="en">    <head>        <!-- ... -->            {% if debug %}            <link rel="stylesheet/less" type="text/css" media="all" href="{{ STATIC_URL }}less/theme.less" />            <script src="{{ STATIC_URL }}js/less-1.3.1.min.js"></script>            {% else %}            {% compress css %}            <link rel="stylesheet" type="text/less" media="all" href="{{ STATIC_URL }}less/theme.less" />            {% endcompress %}            {% endif %}

</p>

And now, we should be good to go.

</p>

* * * * *

</p>

Twitter link that goes at the bottom of a blog post I will put
[here][1].

</p>

<div id="disqus_thread">
</div>
</p>

<p>
<noscript>
Please enable JavaScript to view the [comments powered by Disqus.][]

</noscript>
</p>
[comments powered by <span class="logo-disqus">Disqus</span>][]

  [Bootstrap]: http://twitter.github.com/bootstrap/
  [ways]: https://groups.google.com/forum/?fromgroups#!topic/twitter-bootstrap/GMMmF_nHEiI
  [friends]: http://stackoverflow.com/questions/10451317/twitter-bootstrap-customization-best-practices
  [here]: http://lesscss.org/
  [Fabric]: http://fabfile.org
  [Git tag]: http://git-scm.com/book/en/Git-Basics-Tagging
  [this]: https://gist.github.com/1242040
  [Julia Elman]: http://www.caktusgroup.com/blog/2012/03/05/using-less-django/
  [Django Compressor]: https://github.com/jezdez/django_compressor
  [1]: http://twitter.com/lexandstuff
  [comments powered by Disqus.]: http://disqus.com/?ref_noscript
  [comments powered by <span class="logo-disqus">Disqus</span>]: http://disqus.com
