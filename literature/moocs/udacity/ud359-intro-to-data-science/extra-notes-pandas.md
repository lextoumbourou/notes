# pandas notes

## GroupBy primer (http://wesmckinney.com/blog/?p=125)

* Goals may be
    * Perform aggregation like computing sum of mean of each group.
    * Slice DataFrame into chunks and doing something with each chunk
    * Perform a transformation, like standardizing each group (getting zscore)
* Two main tasks
    * Grouping data
    * Doing something with grouping
* For grouping, define some kind of mapping that assigns labels into group buckets
* A mapping can be
    * A Python function to be called on each label
    * A dict, containing ```{label: group_name}``` mapping
    * An array the same length as the axis containing group correspondencese
* Example, to group data with a data as Index by year
```
> grouped = df.groupby(lambda x: x.year)
```
    * That should return a ```GroupBy``` object
        * Can iterate over it:
        ```
        > for year in group in grouped: print year, group
        ```
        * Can **aggregate** it:
        ```
        > grouped.aggregate(np.sum)
        ```
        * Can **transform groups** like:
        ```
        > zscore = lambda x: (x - x.mean()) / x.std()
        > transformed = grouped.transform(zscore)
        ```
* Can groupby a single column
```
> grouped = df.groupby('A')['C'].mean()
```
