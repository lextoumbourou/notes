# Exercise 4: Introducting Valgrind

* Valgrind is used for memory debugging, memory leak detection and profiling
```
> valgrind ./ext4
```
* Example output:
```
==1308== HEAP SUMMARY:
==1308==     in use at exit: 0 bytes in 0 blocks
==1308==   total heap usage: 0 allocs, 0 frees, 0 bytes allocated
==1308==
==1308== All heap blocks were freed -- no leaks are possible
==1308==
==1308== For counts of detected and suppressed errors, rerun with: -v
==1308== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
```
