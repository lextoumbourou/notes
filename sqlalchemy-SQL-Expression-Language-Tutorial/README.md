SQLAlchemy - notes from docs

* Documentation separated into:
    * ORM
    * Core
    * Dialects
    
## SQL Expression Language Tutorial

### Connecting    

* Return value of ```create_engine``` is an instance of ```Engine```
* First time ```Engine.connect()``` or ```Engine.execute()``` is called, we establish a real DBAPI connection to the database
    
### Define and Create Tables

* Column is usually represented by object called ```Column```
* Table object is called ```Table```
* Define tables within a catalog called ```MetaData```
```
In [6]: from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey

In [7]: metadata = MetaData()

In [8]: users = Table('users', metadata,
   ...:    Column('id', Integer, primary_key=True),
   ...:    Column('name', String),
   ...:    Column('fullname', String)
   ...: )

In [9]: addresses = Table('addresses', metadata,
   ...:    Column('id', Integer, primary_key=True),
   ...:    Column('user_id', None, ForeignKey('users.id')),
   ...:    Column('email_address', String, nullable=False)
   ...: )

```
* Then use ```create_all(engine)``` to actually create the tables. The method is idempotent and can be called multiple times with impunity.

### Insert Expressions

* ```Insert``` construct, represents an insert statement
    * Get ```str``` representation to see SQL generated
    
    ```
    In [15]: ins = users.insert()

    In [16]: str(ins)
    Out[16]: 'INSERT INTO users (id, name, fullname) VALUES (:id, :name, :fullname)'
    ```
    * Use ```values()``` to limit insert statement to certain values
    ```
    In [18]: ins = users.insert().values(name='bill', fullname='jacky hacky')

    In [19]: str(ins)
    Out[19]: 'INSERT INTO users (name, fullname) VALUES (:name, :fullname)'

    In [20]: ins.compile().params
    Out[20]: {'fullname': 'jacky hacky', 'name': 'bill'}
    ```
    
### Executing

* To connect to a database, use the ```engine.connect()``` method. It returns an instance of ```Connection```
* The returned object, has a ```connect()``` method which you can pass a ```ins``` object into.
* This returns a ```ResultProxy``` object.
```
In [27]: type(result)
Out[27]: sqlalchemy.engine.result.ResultProxy
```
* This includes an ```result.inserted_primary_key``` attribute which you can use to get the key of the inserted object

### Executing Multiple Statements

* Can pass in a instance of ```Insert``` into ```conn.execute``` to insert in one line.
* Passing a list of dicts as second param allows you to perform multiple inserts, like so:
```
In [32]: conn.execute(users.insert(), [
   {'name': 'john', 'fullname': 'john magoo'},
   {'name': 'bob', 'fullname': 'bobo'}
])
```

### Selecting

* Use ```select``` function which returns an instance of ```sqlalchemy.sql.selectable.Select``` which can be passed into ```conn.execute```
* This returns a ```ResultProxy``` which can be iterated through like so:
```
In [37]: for r in result: print r
(1, u'bill', u'jacky hacky')
(2, u'john', u'john p')
(3, u'lex', u'Hello')
(4, u'john', u'john magoo')
(5, u'bob', u'bobo')
```
* Can access the values using array-indexes, dict keys or attributes, as follows:
```
In [41]: result = conn.execute(s).fetchone()
In [42]: result['name']
Out[42]: u'bill'

In [43]: result.name
Out[43]: u'bill'

In [44]: result[1]
Out[44]: u'bill'
```
* Call ```result.close()``` to return connection to connection pool.
* Use ```.where()``` called against the ```Select``` object to add filter expressions

### Operators

* Using Python operators again ```Column``` objects, will generator SQL equivalent:
```
In [45]: print users.c.id == addresses.c.id
users.id = addresses.id

In [47]: print users.c.id != None
users.id IS NOT NULL

In [48]: print users.c.id == None
users.id IS NULL
```
* Can generator operators using ```.op('OPERATOR')(value) method```

### Conjunctions

* Import from ```sqlalchemy.sql``` to use
```
In [59]: from sqlalchemy.sql import and_, or_, not_

In [60]: print or_(users.c.id < 100, users.c.id > 2)
users.id < :id_1 OR users.id > :id_2

In [61]: print and_(users.c.id < 100, users.c.id.like('%l'))
users.id < :id_1 AND users.id LIKE :id_2
```
* The ```where``` method can be chained

### Using Joins

* Aside from manually performing ```joins```, they can also be performed with the ```join``` method. 
