# Understanding recursive PostGres queries

For some data relationship problems, there's no escaping recursive PostGres queries. For example, representing a tree-like structure where each node has points to a parent id until a top-level parent. As usual, the documentation is very comprehensive, however, it can be a little tricky to wrap your head around at first. This blog will serve as an attempt to help with the intuition. Hope it helps.

## 99 Bottles of Beer

Firstly, the ``VALUES`` query let's you return a single row with a constant value. For example, ``VALUES (99)`` SQL query returns a table with one row: ``99``. Like this:

```
# VALUES (99);
 column1
---------
       99 
(1 row)
```

The ``WITH`` query builds a temporary table which can be used in subsequent queries. So I can rewrite the above query like this:

```
WITH 99_table AS (
   VALUES (99)
)
SELECT * FROM 99_table;
 column1
---------
       99     
(1 row)
```

Okay, not particularly useful so far, but bear with me. So what if I wanted to perform an operation on the values row in a loop. That's where ``WITH RECURSIVE`` comes in. It's called recursive, but it's really not. It's more analgous to a loop. It's broken down into 3 parts:

1. The base query.
2. The union operation, which defines whether to keep or chuck out duplicate rows (more on this).
3. The recursive query, which will keep operating on the base query until it returns no rows, or hits a ``WHERE`` condition.

Let's see if we can write a simple query that starts with our 99 value and counts down to 1.

```
WITH RECURSIVE 99_table(n) AS (
    VALUES (99)
  UNION ALL
    SELECT n - 1 FROM 99_table WHERE n > 1
)
SELECT * FROM return_one;
```

Cool, so broken down:

1. ``VALUES (99)``: Start with the base query: ``VALUES (99)``.
2. ``UNION ALL``: Keep "all"; eg don't discard duplicate rows.
3. ``SELECT n - 1 FROM 99_table WHERE n > 1``: Replace ``n`` with ``n - 1`` while ``n`` is greater than 1.

Finally, let's do the bottles of beer song:

```
WITH RECURSIVE t(n) AS (
    VALUES (99)
  UNION ALL
    SELECT n - 1 FROM t WHERE n > 1
)
SELECT n::text || ' bottles of beer on the wall' from t;
```

## Walking a tree

Let's try a real world example. Let's say you have some Categories. Then each category can have sub categories and each sub category can have parent categories.

Let's start by creating that table.

```
CREATE TABLE category (
    id integer,
    name varchar(256),
    parent_id integer,
    PRIMARY KEY(id)
);
```

Next, let's add a top level category ``shoes`` and some sub categories.

```
INSERT INTO category (id, name, parent_id) VALUES (1, 'Shoes', NULL);
INSERT INTO category (id, name, parent_id) VALUES (2, 'Sports', 1);
INSERT INTO category (id, name, parent_id) VALUES (3, 'Rebook Pumps', 2);
```

Challenge number 1. Starting with Rebook Pumps, can we find all parent categories?

Okay, so I guess we first define the non-recursive term. So that would be the ``Rebook Pumps``, right?

```
SELECT id, name, parent_id FROM categories WHERE id = 3;
```

Cool. Done. Now I guess we define the recursive bit? I guess we select make our recursive query which should keep trying to join on the intermediate table until there's nothing to join (I think?)

```
WITH RECURSIVE start_category AS (
    SELECT id, name, parent_id
        FROM category WHERE id = 3
    UNION
    SELECT category.id, category.name, category.parent_id
        FROM category
        JOIN start_category ON (category.id = start_category.parent_id)
)
SELECT * FROM start_category;
 id |     name     | parent_id
----+--------------+-----------
  3 | Rebook Pumps |         2
  2 | Sports       |         1
  1 | Shoes        |
(3 rows)
```

Cool. That seems to work. Okay, let's go the other way, start at the top-level and return all the children.

```
WITH RECURSIVE top_level AS (
    SELECT id, name, parent_id
        FROM category WHERE id = 1
    UNION ALL
    SELECT category.id, category.name, category.parent_id
         FROM category
         JOIN top_level ON (category.parent_id = top_level.id)
)
SELECT * FROM top_level;
```

Okay, so going back to the first example what would happen if someone goofed and our top-level category pointed to a low-level category. Effectively creating a circulare relationship. Like this:


Let's try it:

```
UPDATE category SET parent_id = 3 WHERE id = 1;
```

Then run the same query:

```
WITH RECURSIVE start_category AS (
    SELECT id, name, parent_id
        FROM category WHERE id = 3
    UNION ALL
    SELECT category.id, category.name, category.parent_id
        FROM category
        JOIN start_category ON (category.id = start_category.parent_id)
)
SELECT * FROM start_category;
```

Notice how the query never stops? That's because each time the recursive part of the query is run, rows are returned. So it keeps going.

We can combat this problem by using ``UNION`` instead of ``UNION ALL`` which discards duplicate rows:

```
WITH RECURSIVE start_category AS (
    SELECT id, name, parent_id
        FROM category WHERE id = 3
    UNION
    SELECT category.id, category.name, category.parent_id
        FROM category
        JOIN start_category ON (category.id = start_category.parent_id)
)
SELECT * FROM start_category;
 id |     name     | parent_id
----+--------------+-----------
  3 | Rebook Pumps |         2
  2 | Sports       |         1
  1 | Shoes        |         3
(3 rows)
```

Cool, huh?

### How it works.

Okay, so let's try to understand what's going on here, according to the docs.

The first part is to evaluate the non-recursive term. It's then put into a "working" table and what's returned is appended to our results. So we have effectively 2 data structres:

```
SELECT id, name, parent_id FROM category WHERE id = 3;
```

```
Results                               Working table

3   rebook pumps   2                  3  rebook pumps 2
```

Then we evaluate the recursive term against the working table, which in this case is ``SELECT category.id, category.name, category.parent_id FROM category JOIN start_category ON (category.id = start_category.parent_id)``. So basically, get something from ``category`` which can be joined against our working table. So, we fetch the sports category and this time add it to a "Intermediate table".

Intermediate table:

```
id  |   name           | parent_id
----+--------------+-----------
 2  |   Sports         | 1
```

Then this table replaces and working table and the result is appended to results. So we look like this:

```
Results                                Working table

id  |   name           | parent_id       id  |   name           | parent_id
----+--------------------------        -------------------------------------
 3  |   Rebook Pumps   | 2                2  |   Sports         | 1
 2  |   Sports         | 1             
```

Then we go again until nothing is in the working table.
