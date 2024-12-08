---
title: Unicode
date: 2024-12-07
modified: 2024-12-07
status: draft
tags:
  - CharacterEncoding
---

Unicode is a universal character encoding standard designed to support every writing system in the world. At its core, Unicode assigns a unique numerical value (called a code point) to each character across all supported writing systems, similar to how [ASCII](ascii.md) works but with a much broader scope.

## Character Encoding

To store Unicode characters as bytes in computer memory or files, we need encoding rules that translate Unicode strings into byte sequences. The Unicode standard defines several encoding methods, with the most common being:

- [UTF-8](utf-8.md) (Unicode Transformation Format - 8-bit)
- [UTF-16](utf-16.md) (Unicode Transformation Format - 16-bit)
- [UTF-32](utf-32.md) (Unicode Transformation Format - 32-bit)

## UTF-8 Encoding

UTF-8 is the most widely used Unicode encoding method. It's particularly efficient for handling ASCII text while supporting the full Unicode character set. UTF-8 uses a variable-width encoding scheme, meaning different numbers of bytes can represent characters.

### UTF-8 Encoding Rules

1. For code points < 128 (ASCII range):
   - Character is represented by a single byte with the same value
   - Maintains backward compatibility with ASCII

2. For code points between 128 and 0x7FF:
   - Character is encoded as a two-byte sequence
   - Both bytes have values between 128 and 255

3. For larger code points:
   - Characters can be encoded using three or four bytes
   - This allows UTF-8 to represent the entire Unicode range

UTF-8's variable-width design makes it space-efficient while ensuring compatibility with ASCII text, which has helped make it the dominant encoding standard for the web and many other applications.