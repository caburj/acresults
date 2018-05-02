Description
===========

This library contains functions to post-process results of AquaCrop simulations.

Models
------

The AquaCrop result has been modelled in the following way:

**`ACResult`**
    Summary : `Run` object
    Clim : `Result` object
    Inet : `Result` object
    ...

**`ACResult`**
    *methods* To get the `Run` and `Result` object, public methods are used:
    get_summary()
    get(name) where name is a string similar to ['Clim', 'Inet', ...]

**`Run` object**
    *methods*
    get_df()
    get_names()
    get_units()
    get_descriptions()
    get_column(name) where name is string and case insensitive
        *properties*
        .name
        .unit
        .description


**`Result` object**
    *methods*
    get_runs()
    get_run(number) where number is an int starting from 1
    get_units()
    get_names()
    get_descriptions()
    get_column(name) where name is a string and case insensitive.
        Almost similar to get_runs() but only give a particular column.

Functions
---------

`NIR_chart_month`

`boxplot`

`tsplot`
