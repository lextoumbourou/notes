Title: Implementing and customising Lazy Signup in Django
Slug: implementing-and-customising-lazy-signup-in-django
Date: 2012-10-21

In an internet saturated with new things, if you expect anybody to give
your new web application a glance, then you'd better make the barrier to
entry pretty damn low. Hence, Lazy Signup. If you're new to Lazy Sign
up, then this [video][] should help.

</p>

For Django users, [Dan Fair][]'s [django-lazysignup][] is "a package
designed to allow users to interact with a site as if they were
authenticated users, but without signing up."

It works by creating cookie-bound temporary accounts for new users,
allowing them to interact with the site as if they were registered.
Then, when they're ready to sign up, the package provides an overridable
view that allows for a "conversion" to a real account.

In this short tutorial, I'll show you how I setup *django-lazysignup*
for a simple to-do app I'm putting together that allows you to only
create [two tasks per day][] (shut up, my mum thinks it's a good idea).

</p>

### The setup

</p>

My Django project had just one app called *tasks* which provided an
Ajax-enabled template where the users could add, edit and complete
tasks. On the file system, it looks a little something like this:

</p>

<p>
    lex@server:~/twotasks> tree -I "*pyc" -A.+-- __init__.py+-- settings.py+-- tasks¦   +-- __init__.py¦   +-- models.py¦   +-- templates¦   ¦   +-- tasks.html¦   +-- tests.py¦   +-- urls.py¦   +-- views.py+-- templates¦   +-- parent.html+-- urls.py

</p>

### Installing and configuring django-lazysignup

</p>

As per the [docs][], *django-lazysignup* can be installed with Pip.

</p>

<p>
    lex@server:~/twotasks> pip install django-lazysignup

</p>

Next, I added *lazysignup* to the *INSTALLED\_APPS* section of my
settings.py file. Also, I ensured that*django.contrib.auth* was
installed so I can use Django's built-in user authentication system.

Lastly, I added the *lazysignup* authentication backend to the
*AUTHENTICATION\_BACKENDS* section.

</p>

##### settings.py

</p>
<p>
    #...INSTALLED_APPS = (    'django.contrib.auth',    'django.contrib.contenttypes',    'django.contrib.sessions',    'django.contrib.sites',    'django.contrib.messages',    'django.contrib.staticfiles',    'django.contrib.admin',    'lazysignup',    'justtwotasks.tasks',)

</p>

<p>
    AUTHENTICATION_BACKENDS = (    'django.contrib.auth.backends.ModelBackend',    'lazysignup.backends.LazySignupBackend',)

</p>

Then, I ran *syncdb* to setup the tables *lazysignup* uses.

</p>

<p>
    lex@server:~/twotasks> python manage.py syncdb

</p>

Which runs these queries:

</p>

<p>
    lex@server:~/twotasks> python manage.py sqlall lazysignupBEGIN;CREATE TABLE "lazysignup_lazyuser" (    "id" serial NOT NULL PRIMARY KEY,    "user_id" integer NOT NULL UNIQUE REFERENCES "auth_user" ("id") DEFERRABLE INITIALLY DEFERRED,    "created" timestamp with time zone NOT NULL);CREATE INDEX "lazysignup_lazyuser_created" ON "lazysignup_lazyuser" ("created");COMMIT;

</p>

Finally, I created a route to via the */convert* url to the lazysignup
views in my *URLconf*.

</p>

##### urls.py

</p>
<p>
    urlpatterns = patterns('',    #...    url(r'^convert', include('lazysignup.urls')),    #...)

</p>

### Utilising django-lazysignup

</p>

#### Views

</p>

When a user is routed to your view wrapped in the *@allow\_lazy\_user*
decorator, a temporary user account is created for them if they aren't
already signed in.

</p>

In my app, I've got three views, one for displaying the main page, the
other two for handling task related functionality like editing tasks,
deleting tasks etc. All of them should be accessible to the lazy user,
so I applied to decorator to each.

</p>

##### tasks/views.py

</p>
<p>
    from lazysignup.decorators import allow_lazy_user

</p>

<p>
    @allow_lazy_userdef main(request):    """Handles displaying the view to the user"""#...@allow_lazy_userdef add_or_update_task(request):    """Handles adding a new task"""#...@allow_lazy_userdef delete_task(request):    """Handles deleting tasks"""#...

</p>

Now, if I access one of those views by hitting the main homepage, I can
see in my database that a user account is automatically generated.

</p>

<p>
    lex@server:~/twotasks> python manage.py dbshelljusttwotasks=> SELECT id, username, date_joined FROM auth_user; id |            username            |          date_joined----+--------------------------------+-------------------------------  2 | dced705135d642dd8079c945276dea | 2012-10-21 22:58:05.640585+11  1 | lex                            | 2012-10-06 12:34:33.243674+10(2 rows)

</p>

And, we can see the user is defined as a *lazyuser* in the
*lazysignup\_lazyuser* table too.

</p>

<p>
    justtwotasks=> SELECT * from lazysignup_lazyuser; id | user_id |           created----+---------+------------------------------ 1  |      2  | 2012-10-21 22:58:05.65818+11(1 row)

</p>

#### Templates

</p>

In my parent template, I can detect whether the user is a lazy user and
provide a *Sign up* or *Login* button if they are, or information about
the logged in account if they're not by utilising the *is\_lazy\_user*
template filter. This requires importing the *lazysignup* template tags.

Initially, the */convert* page will return a *TemplateDoesNotExist*
exception. I'll fix that next.

</p>

##### templates/parent.html

</p>
<p>
    {% load i18n lazysignup_tags %}#...<ul>{% if not user|is_lazy_user %}    <li>Logged in as <a href="/accounts/profile">{{ user.username }}</a></li>    <li><a href="/accounts/logout">Sign out</a></li>{% else %}    <li>        <a href="/convert">Register</a>    </li>    <li>        <a href="/accounts/login/">Sign in</a>    </li></ul>{% endif %}

</p>

### Setting up the convert page

</p>

The /convert page is where the user can convert their temporary account
to the real thing. In my *templates* folder, I copied the *lazysignup*
template directory from the *django-lazysignup* project.

To do it, I cloned a local copy of the repo, but you could just grab the
files from [Github][] or just simply role your own templates using my
example as a guide.

</p>

I'm going to customize the template to inherit from my base template,
but obviously you can make them look sexy however you want.

</p>

<p>
    lex@server:~/src> git clone git://github.com/danfairs/django-lazysignup.git lex@server:~/src> cd django-lazysignup/lazysignup/templates/lex@server:~/src/django-lazysignup/lazysignup/templates/> cp -R lazysignup ~/twotasks/templates/

</p>

##### templates/lazysignup/convert.html

</p>
<p>
    {% extends "parent.html" %}{% load i18n %}{% block title %}    Register{% endblock %}{% block content %}    <form method="post" action="{% url lazysignup_convert %}">        {{ form.as_p }}        {% csrf_token %}        <input type="hidden" name="redirect_to" value="{{ redirect_to }}">        <input class="btn" type="submit" value="{% trans "Submit" %}" />    </form>{% endblock %}

</p>

Now, if I convert our account, I can see the username update in the
*auth\_user* table and disappear from the *lazyuser* table.

</p>

<p>
    lex@server:~/twotasks> python manage.py dbshelljusttwotasks=> SELECT id, username, date_joined FROM auth_user; id |      username      |          date_joined----+--------------------+-------------------------------  2 | i_just_set_this_up | 2012-10-21 22:58:05.640585+11  1 | lex                | 2012-10-06 12:34:33.243674+10(2 rows)justtwotasks=> SELECT * from lazysignup_lazyuser; id | user_id | created----+---------+---------(0 rows)

</p>

### Customising the convert form

</p>

So, now that's working rather nicely. I can create tasks, close my
browser and, providing my session is still active, access the tasks.

Then, when I'm ready, I can create an account and it's all gravy.

But, what happens if I want to capture more than just username and
password in the signup process?

Perhaps, I would also like to capture the user's email address, or full
name etc.

*lazysignup* lets you pass in a custom form to the *convert* view,
provided it matches the specifications listed [here][].

My example extends the *UserCreationForm* class adding email,
first\_name and last\_name fields.

</p>

##### forms.py

</p>
<p>
    from django import formsfrom django.contrib.auth.models import Userfrom django.contrib.auth.forms import UserCreationFormfrom django.utils.translation import ugettext_lazy as _class TwoTasksUserCreationForm(UserCreationForm):    """    Class passed into lazysignup view to override default and allow    for email address username    """    email = forms.EmailField(label=_("Email"),        help_text = _("Required. A valid email address."),        error_messages = {            'invalid': _("That doesn't appear to be a valid email address.")})    first_name = forms.CharField(label=_("First Name"), max_length=30,        help_text = _("Optional."),        error_messages = {            'invalid': _("That doesn't appear to be a valid name.")})    last_name = forms.CharField(label=_("Last Name"), max_length=30,        help_text = _("Optional."),        error_messages = {            'invalid': _("That doesn't appear to be a valid name.")})    class Meta:        model = User        fields = ("username", "password1", "password2", "email",                  "first_name", "last_name")    def get_credentials(self):        return {            "username": self.cleaned_data["username"],            "password": self.cleaned_data["password1"]}

</p>

Now in my *URLconf*, I can pass in my freshly created form to the
convert view using the *form\_class* parameter.

</p>

##### urls.py

</p>
<p>
    from justtwotasks.forms import TwoTasksUserCreationForm as Form

</p>

<p>
    urlpatterns = patterns('',    #...    url(r'^convert', include('lazysignup.urls')), {'form_class' : Form}),    #...)

</p>

### That's a wrap.

</p>

Hopefully that helped you get up and running with *django-lazysignup*.

Feel free to [let me know][] if there's any issues with my tutorial.

Just as long as you don't follow me on [Twitter][].

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

  [video]: http://www.youtube.com/watch?v=uBT2niL2jKM
  [Dan Fair]: https://github.com/danfairs/
  [django-lazysignup]: https://github.com/danfairs/django-lazysignup
  [two tasks per day]: http://www.justtwotasks.com
  [docs]: http://django-lazysignup.readthedocs.org/
  [Github]: https://github.com/danfairs/django-lazysignup/tree/master/lazysignup/templates
  [here]: http://django-lazysignup.readthedocs.org/en/latest/usage.html#the-converted-signal
  [let me know]: mailto:lextoumbourou@gmail.com
  [Twitter]: http://twitter.com/lexandstuff
  [comments powered by Disqus.]: http://disqus.com/?ref_noscript
  [comments powered by <span class="logo-disqus">Disqus</span>]: http://disqus.com
