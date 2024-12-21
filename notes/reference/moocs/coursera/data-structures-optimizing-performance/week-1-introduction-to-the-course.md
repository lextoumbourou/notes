---
title: "Week 1 - Introduction to the Course"
date: 2016-10-04 00:00
category: reference/moocs
status: draft
parent: data-structures-optimizing-performance
---

* [Flesch Reading Ease](../../../../../../permanent/flesch-reading-ease.md)
    * Score is a measure of text readability
        * Formula: `206.835 - 1.015 * (# words / # sentences) - 84.6 * (# syllables / # words)`
        
* [Interned Strings](../../../../../../permanent/interned-strings.md)
    * Allow the language to treat 2 duplicate strings as the same object in memory.

```java
String text = new String("Text shiz");  # New object
String text2 = "Hello world!";  # Refers to the same "interned" object.
String text3 = "Hello world";  # Refers to the same "interned" object.
```

* String methods in Java:
    * ``length``
    * ``toCharArray``
    * ``charAt`` - return character at string position
    * ``split``
    * ``indexOf``

* For each loop in Java:

    ```java
    for (char c : word.toCharArray())
    {
      if (c == letter)
      {
          return true;
      }
    }
    ```

* Regular Expressions primer
    * ``a`` - match ``a``.
    * ``a+`` - match one or more ``a``s.
    * ``a*`` - match zero or more ``a``s.
    * ``(ab)+`` - match one or more ``ab``s.
    * ``[abc]`` - match any character inside the set.
    * ``[a-c]`` - match any character between ``a`` and ``c``.
    * ``[^a-c]`` - match anything that's not between ``a`` and ``c``.
    * ``a|c`` - match ``a`` or ``c``.