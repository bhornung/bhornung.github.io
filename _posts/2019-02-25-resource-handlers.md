---
layout: post
title:  "Resource handlers"
date:   2019-02-25 21:25:44 +0000
categories: python
---


We create a source agnostic interface that yields data to a pipeline.


## Motivation

Data can be loaded from a variety of sources, files on hard drive, from a database, as a stream from a sensor. Each of these resources require a different set of rules to extract the data from them. When constructing a data pipeline, we can easily end up writing custom handler, each being slightly different, depending on the project we are working on. Even worse, the subsequent stages of the data pipeline would explicitly depend on the specifications of the handler. 

A common solution is to write interfaces with highly abstracted functionalities, where an interface
* belongs to a program unit that performs a generic task _i.e._ consuming the contents of the resource
* has a standardised and preferably intuitive signature.

## Specifications

Before we set out to write a handler, it is important to define a set of requirements that will guide us throughout the implementation. Having the specifications will also help us to quantify the work and progress that we will have done.

In the following we create a data handler which has the following methods

* `open_resource` opens and connects to a resource
* `get_one` returns one unit of data
* `get_all` returns all data units
* `close_resource` disconnects from and closes the resource

The raw notebook can be viewed [here](https://github.com/bhornung/bhornung.github.io/blob/master/assets/resource-handlers/notebook/resource-handlers.ipynb).

### Mock-up class

We create a class called `DummyResourceHandler` which serves as a mockup for the future classes.


```python
class DummyResourceHandler:
    
    def __init__(self, *args, **kwargs):
        """
        Creates an istance of the data handler.
        """
        pass
    
    def open_resource(self, *args, **kwargs):
        """
        Opens and connects to a resource, so that data can be consumed from it.
        """
        pass
    
    def close_resource(self, *args, **kwargs):
        """
        Disconnects from and closes the resource. It also performs clean up actions if required.
        """
        pass
    
    def get_one(self):
        """
        Retrieves one block of data from the resource.
        """
        pass
    
    def get_all(self, *args, **kwargs):
        """
        Retrieves all data blocks.
        """
        pass
```

The class above might seem to of limited practical use. Its methods are empty, however three important decision have been made

* we assigned names to the methods
* specified in the docstrings what each method is supposed to do 
* specfied the signature of each method
* we also made really important design decision, that the initialisation of class does not result in opening or connecting to the resource. 

## Implementation

Ideally, we wish to create something that is usable. Depending on the level of abstraction[, [time constraints[, and mood], there are three different routes to take.

### 1. Customised class

A customised class contains methods that are custom written for specific kind of resource. In other words, a standalone class is created for each type of resource, whilst keeping the method names and signatures. 

The `FileHandler1` class serves as an example of the custom made handler.


```python
class FileHandler1:
    
    def __init__(self, raise_on_end = True):
        """
        Initialises the handler.
        Parameters:
            raise_on_end (bool) : whether to raise an error if one tries to call
            get_one once the resource has been exhausted. Default: True.
        """
        
        self._is_resource_open = False
        
        self._raise_on_end = raise_on_end
        
    def open_resource(self, path_to_file, file_type = 'text'):
        """
        Opens a file for reading.
        Parameters:
            path_to_file (str) : full path to file to be read
            mode (str) : whether text of binary file
        """
        
        # attach resource
        self._resource = slim_file_reader(path_to_file, file_type = file_type)
        self._data_generator = (x for x in self._resource)
        self._is_resource_open = True
        
    def get_all(self):
        """
        Yields all records in a file.
        Returns:
            data_generator (generator) : a generator that iterates over the resource.
        """
    
        if not self._is_resource_open:
            raise IOError("Resource has not been opened.")
         
        data_generator = (x for x in self._resource)
        
        return data_generator
    
    def get_one(self):
        """
        Returns one datum from resource.
        Please note, the resource is exhausted it returns None.
        """
    
        if not self._is_resource_open:
            raise IOError("Resource has not been opened.")        
        
        try:
            return next(self._resource)
        
        # what to do when the iterator is exhausted
        except StopIteration:
            if self._raise_on_end:
                raise
            else:
                return None
        
        # unexpected error reraised
        except:
            raise
            
    def close_resource(self):
        """
        Empty method. 
        The underlying resource is a file. Its context manager takes care of cloding it safely.
        """
        pass
```

#### Trial

In the first example a [file](https://github.com/bhornung/bhornung.github.io/blob/master/assets/online-encoder/data/trial.txt) is read line by line:


```python
handler1 = FileHandler1()
handler1.open_resource('trial.txt')

print("Consume all lines from file:")
for idx, line in enumerate(handler1.get_all()):
    print("{0}. line: {1}".format(idx, line.strip()))
```

    Consume all lines from file:
    0. line: first
    1. line: 1
    2. line: 2
    3. line: 3
    4. line: 4
    

In the second example, the bytes are read. We will produce `None` once all bytes are cycled through:


```python
handler2 = FileHandler1(raise_on_end = False)
handler2.open_resource('trial.txt', file_type = 'binary')

print("Sequentially consume bytes from file:")
print(" | ".join((str(handler2.get_one()) for idx in range(20))))
```

    Sequentially consume bytes from file:
    b'f' | b'i' | b'r' | b's' | b't' | b'\r' | b'\n' | b'1' | b'\r' | b'\n' | b'2' | b'\r' | b'\n' | b'3' | b'\r' | b'\n' | b'4' | b'' | None | None
    

#### Analysis

A number of remarks are in order.

* Using separate constructor and opener, neatly divides the concerns of each methods. The arguments passed to the constructor affects the general behaviour of the class, whereas arguments passed to the `open_resource` function, defines how the resource should be handled.
    * The `get_one` method can either raise an error or return None if EOF is reached. The actual behaviour is set by the `raise_on_end` switch passed to the constructor.
    * The `open` method decides how wehther to treat the file as binary of text.
* The `close` method is empty for the open context manager closes the file when it finds appropriate
* There is a mild danger however. If one uses `get_one` and then `get_all`, the former method only consumes the what is left of the generator, but not the entire file. It would be easy to fix this by calling `open` again from inside `get_all`. However, we keep the class simple for demonstration purposes.

### 2. Abstract base class

Writing a custom class for each scenario can be tedious, one has to keep in mind and adhere to all the conventions.  It is therefore advantageous to create a scheme that will be enforced.

Creating custom classes might as well result in code repetition, a prime soil for raising errors. 

Thirdly, some components of the class can be preserved. For example, if we inspect the `get_one` and `get_all` methods, we quickly realise they are oblivious of exact identity of the underlying `_resource` field. It only has to support iteration. 

#### Aside

It would be possible to ignore the `open_resource` method altogether and pass an object to the constructor, then create a generator of the resource:

```
    ...
    def __init__(self, resource):
    
        self._data_generator = (x for x in resource)
    ...
```

This solution might seem neater, _e.g._ one can pass a `numpy` array to the constructor. However, this is **not** what we set out to do above. We aim to create a handler that can open a resource. Had we introduced the modification above, an already opened file handler would have been necessary to be passed to the constructor. There is always a tradeoff...

A solution is to create an abstract baseclass from which all handlers can be derived. Failing to implement any of the methods decorated by `abc.abastractmethod` would result in a `TypeError` during runtime. The `ResourceHandlerBaseClass` below serves as a template for the handlers.


```python
class ResourceHandlerBaseclass(abc.ABC):
    
    def __init__(self, *args, **kwargs):
        
        self._is_resource_open = False
    
    @abc.abstractmethod
    def open_resource(self, *args, **kwargs):
        """
        Opens and connects to a resource, so that data can be consumed from it.
        """
        pass
    
    @abc.abstractmethod
    def close_resource(self, *args, **kwargs):
        """
        Disconnects from and closes the resource. It also performs clean up actions if required.
        """
        pass
    
    @abc.abstractmethod
    def get_one(self):
        """
        Retrieves one block of data from the resource.
        """
        pass
    
    @abc.abstractmethod
    def get_all(self, *args, **kwargs):
        """
        Retrieves all data blocks.
        """
        pass
```

We can now proceed to specifiy the methods. In this example, `FileHandler2`, a datum corresponds to a specified number of subsequent lines from a file. (Please note some of the docstrings are omitted for sake of brevity.)


```python
class FileHandler2(ResourceHandlerBaseclass):
    
    def __init__(self, raise_on_end = True):
        
        self._is_resource_open = False
        
        self._raise_on_end = raise_on_end
        
    def open_resource(self, path_to_file, n_lines_in_block = 10):
        
        # attach resource
        self._resource = read_nlines_from_file(path_to_file, n_lines_in_block = n_lines_in_block)
        self._data_generator = (x for x in self._resource)
        self._is_resource_open = True
        
    def get_all(self):
        """
        Yields all n-line blocks in a file.
        Returns:
            data_generator (generator) : a generator that iterates over the resource.
        """
    
        if not self._is_resource_open:
            raise IOError("Resource has not been opened.")
         
        data_generator = (x for x in self._resource)
        
        return data_generator
    
    def get_one(self):
        """
        Returns one n-line block from a file
        """
    
        if not self._is_resource_open:
            raise IOError("Resource has not been opened.")        
        
        try:
            return next(self._resource)
        
        # what to do when the iterator is exhausted
        except StopIteration:
            if self._raise_on_end:
                raise
            else:
                return None
        
        # unexpected error reraised
        except:
            raise
            
    def close_resource(self):

        pass        
```

#### Trial

Blocks of three line length are consumed from a text file:


```python
handler3 = FileHandler2()
handler3.open_resource('trial.txt', n_lines_in_block = 3)

for idx, block in enumerate(handler3.get_all()):
    print("{0}. block: {1}".format(idx, block))
```

    0. block: ['first\n', '1\n', '2\n']
    1. block: ['3\n', '4']
    

### 3. Abstract class with defined methods

We have already noticed that the `get_one` and `get_all` methods are conserved. We can thus incorporate them in the base class. 

* The inherited classes are still required to implement them explicitly. 
* However, one can revert to parent's methods by calling _e.g._ `super().get_one()`
* Both methods access the `_resource` property. It is initialised as an empty generator in the constructor. Alternatively, it can be defined as an abstract property.

These changes are implemented in the `ResourceHandlerBaseClass2` class below.


```python
class ResourceHandlerBaseclass2(abc.ABC):
    
    def __init__(self, *args, **kwargs):
        
        self._is_resource_open = False
        
        self._raise_on_end = kwargs.get('raise_on_end', True)
        
        self._resource = (x for x in [])
        
        # add any custom code here
    
    @abc.abstractmethod
    def open_resource(self, *args, **kwargs):
        """
        Opens and connects to a resource, so that data can be consumed from it.
        """
        pass
    
    @abc.abstractmethod
    def close_resource(self, *args, **kwargs):
        """
        Disconnects from and closes the resource. It also performs clean up actions if required.
        """
        pass
    
    @abc.abstractmethod
    def get_all(self):
        """
        Yields all records in a file.
        Returns:
            data_generator (generator) : a generator that iterates over the resource.
        """
    
        if not self._is_resource_open:
            raise IOError("Resource has not been opened.")
         
        data_generator = (x for x in self._resource)
        
        return data_generator
    
    @abc.abstractmethod
    def get_one(self):
        """
        Returns one datum from resource.
        Please note, the resource is exhausted it returns None.
        """
    
        if not self._is_resource_open:
            raise IOError("Resource has not been opened.")        
        
        try:
            return next(self._resource)
        
        # what to do when the iterator is exhausted
        except StopIteration:
            if self._raise_on_end:
                raise
            else:
                return None
        
        # unexpected error reraised
        except:
            raise
```

We proceed to create a handler class that recieves its data from a buffered resource. This resource, in real life, can be the output of a crawler, or results of repeated queries of a database, or a stream of packages from a sensor.

The `open_resource` method tries to connect to a resource by calling its `flush` method. If it is not found, an error is raised.


```python
class BufferedResourceHandler(ResourceHandlerBaseclass2):
    """
    Yields data from a buffered resource.
    """

    def __init__(self):
        super().__init__()
        
    def open_resource(self, resource):
        """
        Connects to resource.
        Parameters:
            resource : must implement the 'flush()' method
        """
    
        try:
             self._resource = resource.flush()
        
        except AttributeError:
            raise
        
        except:
            print("Unexpected error happened.")
            raise
            
        self._is_resource_open = True
        
    def get_one(self):
        return super().get_one()
        
    def get_all(self):
        return super().get_all()
    
    def close_resource(self):
        
        self._resource = None 
```


```python
buffered_resource = BufferedResource()

handler4 = BufferedResourceHandler()
handler4.open_resource(buffered_resource)

print("First fifteen elements:")
print(" | ".join((str(handler4.get_one()) for idx in range(15))))
```

    First fifteen elements:
    0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14
    

#### Analysis

A number of important points are to be written down:

* the `get_one` and `get_all` methods are implemented as direct calls to the corresponding methods of the parent class. We just enforced ourselves to write them out explicitly. Something of an antipattern.
* As an option, the wrappers around `get_one` and `get_all` could be removed in the `ResourceHandlerBaseclass2`, if we are confident the resource field is always the source of the data.
* Strictly speaking, the `open_resource` method does not open the resource, it connects to it. The implications of this are discussed below.

## Summary and prospects


### What have we achieved?

We created a resource handler that 
* serves either a datum by calling the `get_one()` method
* or all data by calling the `get_all()` method

##### Was it worth the time?

Yes, the rest of our application is detached from the source and handling of the incoming data. The only thing we have to remember is to call one of the methods above whenever needed inside our pipeline. The rest of the pipeline is agnostic whence the data sourced. The handler is not interested thence the data go.

### What can be done next?

There are two directions of further abstracting the handler utilities

#### Wrap data source

The signature of the base class, allow us to
* create a data source inside the handler _e.g._ open a file in `FileHandler2`, attach to a database
* connect to a stream as in `BufferedResourceHandler`

If we choose the latter option, it is possible to pass argument to the `open_resource` method which prescribe the way to connect to the stream _e.g._ one can pass keyword arguments which specify which method of the stream should be used.

#### Class factory

Nothing prevents us from writing a class factory that dynamically creates handler classes inherited from the base class. If we work with an established code base, that its utilities can be used as off-the-shelf components to populate the fields of the dynamically created class.

## Auxilliary utilities

This section contains the auxilliary function that were invoked to illustrate various aspects of the handlers.

### File I/O utilities

A slim function, `slim_file_reader`, is created to read a file sequentially. The user can choose to process the file as text or consume the raw bytes. (There are more efficient ways to process a binary file _e.g._ read chunks. We just give this option as to illustrate the flexibility of the class.)


```python
def read_byte_from_file(path_to_file):
    """
    Yields bytes sequentially from a file.
    """
    
    with open(path_to_file, 'rb') as fproc:
        byte = True
        try:
            while byte:
                byte = fproc.read(1)
                yield byte
                
        except StopIteration:
            pass
```


```python
def read_line_from_file(path_to_file):
    """
    Yields lines sequentially from a file.
    """
    
    with open(path_to_file, 'r') as fproc:
        try:
            while True:
                yield next(fproc)
                
        except StopIteration:
            pass 
```


```python
def slim_file_reader(path_to_file, file_type = 'text'):
    """
    Creates an iterator over a file.
    Parameters:
        path_to_file (str) : full path to the file
    """
    
    if file_type.lower() == 'text':
        mode = 'r'
        return read_line_from_file(path_to_file)
        
    elif file_type.lower() == 'binary':
        mode = 'rb'
        return read_byte_from_file(path_to_file)
    
    else:
        raise ValueError("keyword argument 'file_type' must be 'text' or 'binary'. Got: {0}".format(file_type))
```

The `read_nlines_from_file` function creates a block of lines read sequentially from a file.


```python
def read_nlines_from_file(path_to_file, n_lines_in_block = 10):
    """
    Read n subsequent lines from a file.
    Parameters:
        path_to_file (str) : full path to the file
        n (int) : number of lines to in a block
    """
    with open(path_to_file, 'r') as fproc:
        try:
            
            buffer = []
            count = 0
            while True:
                
                while count < n_lines_in_block:
                    buffer.append(next(fproc))
                    count += 1
                
                yield buffer
                buffer = []
                count = 0
                
        except StopIteration:
            
            # unfinished buffer
            yield buffer
            pass
```

### Buffered resource

The `BufferedResource` class serves as a mockup for any resource that serves an infinite stream of data. One can think of an example of a crawler that scrapes tables from linked websites.


```python
class BufferedResource:
    """
    Mockup for an infinite buffered resource. It 
    * gathers data
    * exposes it for consumption
    """
    
    def __init__(self, buffer_size = 10):
        """
        Parameters:
            buffer_size (int) : a positive integer defining how many data are held in the buffer before flushing.
        """
        if not isinstance(buffer_size, int):
            raise TypeError("'buffer_size' must be int. Got: {0}".format(type(buffer_size)))
            
        if buffer_size < 1:
            raise ValueError("'buffer_size' must be larger than zero. Got: {0}".format(buffer_size))
        
        self._buffer_size = buffer_size
        
        self._idx = 0
        
        # initialise with an empty generator
        self._buffer = (x for x in [])
        
    def _load_buffer(self):
        """
        Replenishes the the data. It is a placeholder method.
        """
        
        # range can be replaced by a more elaborate function e.g. crawler
        elements = range(self._idx, self._idx + self._buffer_size)
        self._buffer = (x for x in elements)
        self._idx += self._buffer_size
        
    def flush(self):
        """
        Sequentially yields the contents of the buffer.
        """
        
        while True:
            try:
                yield next(self._buffer)
             
            # if exhausted, replenish data
            except StopIteration:
                self._load_buffer()
```

