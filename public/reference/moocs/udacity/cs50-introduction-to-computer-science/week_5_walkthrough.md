File I/O
========
* Files are a sequence of bytes
* Input: read the bytes in some format
* Output: write the bytes to the file

File Position Indicator
=======================
* Where you are in the file
* fopen: C function to open a file
* fread takes 4 arguments
    * &data: pointer to a struct, which will contain bytes of a file once freed has finished
    * size: size of each element to read 
    * number: number of elements to read
    * inptr: pointer to the file
* fwrite
* fputc (??)
* fseek (inptr, amount, from)
    * from:
        * SEEK_SET (beginning of file)
        * SEEL_CUR (current position in file)
        * SEEK_END (end of file)

Bitmaps
=======
* Each colour represented by 3 bytes (aka scaled from 0 - 255)
    * amount of blue
    * amount of green
    * amount of red

RGB Triples
===========
* pixels are represented by RGBTRIPLE structs
    * RGBTRIPLE triple;
      triple.rgbtBlue = 0x00;
      triple.rgbtGreen = 0x00
      triple.rgbtRed = 0xff

Padding
=======
* size of each scanline must be multiple of 4 bytes (each pixel is 3 bytes)
* if number of pixels per line x 3 is not multiple of 4, we need padding
    * padding is just 0s to make num of bytes be a multiple of 4

Bitmap File Header
==================
* bfSize: total size of image (in bytes), including header, pixels and padding
* biSize: total size of image, inc pixels and padding
* biWidth: width of images, not including padding
* biHeight: height of image

xxd
===
* Unix command-line tool for viewing byte representation of data
* Options:
    * -s - how far to skip
    * -g - how many bytes to read in at a time
    * -c - colums per line

resize
======
* Given some pixel: horizontally resize it and vertically resize it
* Duplicate pixels across
* Duplicate lines down
* Update the header info

Resizing horizontally
=====================
* copy.c reads in a single pixel, then writes a single pixel, instead we read a pixel then write it n times
* padding: old image and new image might have different padding
    * need to recalulate based on formula: 3 bytes per pixel, number of bytes per line must be multiple of 4
    * when reading, using original padding
    * when writing, use newly calculated padding

Resizing vertically
==================
* write each line n - 1 more times

Recover
======
* open card.raw
* determine start of new image
* determine filename
* write bytes of image to the same file

JPEGS
=====
* sequence of byte
* start with either
    0xff 0xd8 0xff 0xe0
    0xff 0xd8 0xff 0xe1
* stored contiguously on the CF card

Writing Blocks
=============
* once we find an image, we can fwrite into file until we find start of another image
* feof (input)
    * returns a boolean
