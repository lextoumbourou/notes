Title: Introducing GiantDwarf
Slug: introducing-giantdwarf
Date: 2012-07-31

I had a couple of hours (somewhat) free this morning so I put together a
little [Campfire][] bot for [Nagios][]. Nothing too fancy, the dwarf
just scrapes the notification page using **urllib2** and
**BeautifulSoup**, and tells us what's going on via the [Campfire API][]
wrapper, [PyFire][]. I used PyFire over [Pinder][] because it uses
**urllib2** as its connection handler, which seems to handle proxy
servers better than **httplib2**.

GiantDwarf warns us when bad things are about to happen...

</p>

![image][]

And tells us when they do...

</p>

![image][1]

He knows when our servers break too...

</p>

![image][2]

But it's okay because we're good at fixing them...

</p>

![image][3]

More at [Github][].

</p>

  [Campfire]: http://campfirenow.com/
  [Nagios]: http://www.nagios.org/
  [Campfire API]: https://github.com/37signals/campfire-api
  [PyFire]: https://github.com/mariano/pyfire
  [Pinder]: https://github.com/rhymes/pinder
  [image]: http://lexandstuff.com/static/lexandstuff/images/articles/giant_dwarf_warning.png
  [1]: http://lexandstuff.com/static/lexandstuff/images/articles/giant_dwarf_critical.png
  [2]: http://lexandstuff.com/static/lexandstuff/images/articles/giant_dwarf_fail.png
  [3]: http://lexandstuff.com/static/lexandstuff/images/articles/giant_dwarf_ok.png
  [Github]: https://github.com/lextoumbourou/GiantDwarf
