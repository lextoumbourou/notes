# Lesson 3: MapReduce Code 

* Input Data
    * Each mapper processes input mapper, each mapper processes a line at a time
    * Input
        ```
        <date> <time> <store-name> <product-type> <cost> <method-of-payment>
        ```
    * Question: Total sales intermediate key?
        * What would be a suitable reducer?
* Basic mapper code:

```
def mapper():
    for line in sys.stdin:
        data = line.strip().split("\t")
        date, time, store, item, cost, payment = data
        print "{0}\t{1}".format(store, cost)
```

* Reducing
    * In the example, data comes in as lines of text
    ```
    def reducer():
        sales_total = 0
        old_key = None
        
        for line in sys.stdin:
            data = line.strip().split("\t")

            if len(data) != 2:
                continue

            this_key, this_sale = data

            if old_key and old_key != this_key:
                print "{0}\t{1}".format(old_key, sales_total)
                sales_total = 0

            old_key = this_key
            sales_total += float(this_sale)

        if old_key != None:
            print "{0}\t{1}".format(old_key, sales_total)
    ```
* Mapper can support input from stdin
    * Type data and press Ctrl + D when finished to test

## Project

* Getting started
    * Follow instructions [here](https://docs.google.com/document/d/1v0zGBZ6EHap-Smsr3x3sGGpDW-54m82kDpPKC2M6uiY/pub)
    * Put VM into Bridged Mode (personal preference)
    * Sample code available under ```~/udacity_training/code/```
    * To see sample code in action
        * Put ```purchases.txt``` onto the HDFS
        ```
        > cd ~/udacity_training/data
        > hadoop fs -put training.txt
        > hs mapper.py reducer.py training.txt output_path
        > hadoop fs -ls output/
        Found 3 items
        -rw-r--r--   1 training supergroup          0 2014-01-31 05:00 output/_SUCCESS
        drwxr-xr-x   - training supergroup          0 2014-01-31 04:58 output/_logs
        -rw-r--r--   1 training supergroup       2296 2014-01-31 05:00 output/part-00000
        > hadoop fs -cat output/part-00000
        ```


  
