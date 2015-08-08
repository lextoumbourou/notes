# Chapter 3: Data in, data out

* Document, in ES, refers to top-level or ``root object`` serialized into JSON.
* Document metadata, consists of 3 fields.
  * ``_index`` - where the document lives.
    * cannot begin with underscore.
    * must be lower case.
    * cannot contain commas.
  * ``_type`` - class of object document represents.
    * has own mapping + schema definition.
    * can't begin with an underscore or contain commas.
  * ``_id`` - unique id for object.
* Indexing a document.
  * Can use automatically generated id (url-safe base64-encoded UUID).
  * Can provide own id.
* Retrieving a document
  * Can retrieve just the source of doc by appending ``_source`` to the url: ``http://127.0.0.1:9200/users/blogs/123/_source``
  * Can check whether doc exists using a ``HEAD`` call: ``curl -i -XHEAD http://127.0.0.1:9200/websites/users/123``
* Updating a whole document
  * Docs are *immutable* ie they can not be updated or modified.
  * All updates reindex whole document.
  * When a new version of a doc is added, it's version is bumped: 
  ```
  curl http://curl http://127.0.0.1:9200/websites/blogs/1286?pretty=1
  {
    "_index" : "inbox-2015-02-18-t21:13:17.290748",
    "_type" : "blogs",
    "_id" : "1286",
    "_version" : 2,
  ```
* Creating a new document
  * Use either ``optype`` query param or append ``_create`` to url. 
    * ```PUT http://127.0.0.1:9200/websites/blog/123?op_type=create```
    * ```PUT http://127.0.0.1:9200/websites/blog/123/_create```
  * Will raise a 201 if successful or 409 if document already exists.
* Deleting a document
  * ```DELETE http://127.0.0.1:9200/websites/blog/123```
  * Response will include a bumped version number.
* Dealing with conflicts
