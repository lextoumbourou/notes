# Python Profiles

* ``cProfile`` - recommended for most users; C extension with reasonable overhead.
* ``profile`` - pure Python version of ``cProfile``, adds significant overhead to profiled programs. 

* Profiling example:

```
In [1]: import cProfile

In [2]: import re

In [3]: cProfile.run('re.compile("foo|bar")')
         195 function calls (190 primitive calls) in 0.000 seconds

   Ordered by: standard name

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.000    0.000 <string>:1(<module>)
        1    0.000    0.000    0.000    0.000 re.py:192(compile)
        1    0.000    0.000    0.000    0.000 re.py:230(_compile)
        1    0.000    0.000    0.000    0.000 sre_compile.py:228(_compile_charset)
        1    0.000    0.000    0.000    0.000 sre_compile.py:256(_optimize_charset)
        1    0.000    0.000    0.000    0.000 sre_compile.py:433(_compile_info)
        2    0.000    0.000    0.000    0.000 sre_compile.py:546(isstring)
        1    0.000    0.000    0.000    0.000 sre_compile.py:552(_code)
        1    0.000    0.000    0.000    0.000 sre_compile.py:567(compile)
      3/1    0.000    0.000    0.000    0.000 sre_compile.py:64(_compile)
        5    0.000    0.000    0.000    0.000 sre_parse.py:137(__len__)
       12    0.000    0.000    0.000    0.000 sre_parse.py:141(__getitem__)
        7    0.000    0.000    0.000    0.000 sre_parse.py:149(append)
      3/1    0.000    0.000    0.000    0.000 sre_parse.py:151(getwidth)
        1    0.000    0.000    0.000    0.000 sre_parse.py:189(__init__)
       10    0.000    0.000    0.000    0.000 sre_parse.py:193(__next)
        2    0.000    0.000    0.000    0.000 sre_parse.py:206(match)
        8    0.000    0.000    0.000    0.000 sre_parse.py:212(get)
        1    0.000    0.000    0.000    0.000 sre_parse.py:317(_parse_sub)
        2    0.000    0.000    0.000    0.000 sre_parse.py:395(_parse)
        1    0.000    0.000    0.000    0.000 sre_parse.py:67(__init__)
        1    0.000    0.000    0.000    0.000 sre_parse.py:706(parse)
        3    0.000    0.000    0.000    0.000 sre_parse.py:92(__init__)
        1    0.000    0.000    0.000    0.000 {_sre.compile}
       15    0.000    0.000    0.000    0.000 {isinstance}
    39/38    0.000    0.000    0.000    0.000 {len}
        2    0.000    0.000    0.000    0.000 {max}
       48    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        5    0.000    0.000    0.000    0.000 {method 'find' of 'bytearray' objects}
        1    0.000    0.000    0.000    0.000 {method 'items' of 'dict' objects}
        8    0.000    0.000    0.000    0.000 {min}
        6    0.000    0.000    0.000    0.000 {ord}
```

* First line indicates that 195 calls were monitored.
* 192 were "primitive" meaning that the call wasn't induced via recursion.
 * ``Order by: standard name``; text at far right was used to sort the output.
* Column headers:
  * ``ncalls`` - number of calls.
    * When there are 2 numbers, like ``3/1``, it means the function is recursive.
      * 2nd value == number of primitive calls.
      * First == total calls.
  * ``tottime`` - total time spent running the given function (excluding time spend in sub functions).
  * ``percall`` - result of ``tottime`` / ``ncalls``
  * ``cumtime`` - time spent in function and subfunctions.
  * ``percall`` - result of ``cumtime`` / primative calls
  * ``filename:lineno(function)`` - respective data of each function.

* Can use ``cProfile`` to invoke another script, like so:

```
python -m cProfile -o dump -s cumtime some_script.py
```

* Then, read the output with ``pstats``:

```
import pstats
p = pstats.Stats('./profile.txt')
p.strip_dirs().sort_stats('cumulative').print_stats(10)
Tue Mar  1 23:06:06 2016    ./dump

         736132 function calls (730253 primitive calls) in 1.616 seconds

   Ordered by: cumulative time
   List reduced from 2866 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    327/1    0.011    0.000    1.617    1.617 {built-in method exec}
        1    0.000    0.000    1.617    1.617 compare.py:1(<module>)
        1    0.006    0.006    1.236    1.236 compare.py:33(runner)
      834    0.006    0.000    1.229    0.001 utils.py:58(_wrapped)
      833    0.003    0.000    1.217    0.001 __init__.py:445(search)
      834    0.014    0.000    1.186    0.001 transport.py:275(perform_request)
      833    0.012    0.000    1.100    0.001 http_urllib3.py:71(perform_request)
      833    0.018    0.000    1.074    0.001 connectionpool.py:435(urlopen)
      833    0.016    0.000    0.888    0.001 connectionpool.py:321(_make_request)
      833    0.005    0.000    0.774    0.001 client.py:1130(getresponse)


Out[4]: <pstats.Stats at 0x1113a2278>
```

* Use ``cumulative`` for figuring out what algorithms are taking time.

* To see what functions are looping a lot, use ``time``:

```
p.strip_dirs().sort_stats('time').print_stats(10)
```
