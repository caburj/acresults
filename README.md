Description
===========

This library contains functions to post-process results of AquaCrop simulations.

Models
------

The AquaCrop result has been modelled in the following way:

**`ACResult`**
- Summary : `Run` object
- Clim : `Result` object
- Inet : `Result` object
- ...

**`ACResult`**
- *methods* To get the `Run` and `Result` object, public methods are used:
- get_summary()
- get(name) where name is a string similar to ['Clim', 'Inet', ...]

**`Run` object**
- *methods*
- get_df()
- get_names()
- get_units()
- get_descriptions()
- get_column(name) where name is string and case insensitive
    - *properties*
    - .name
    - .unit
    - .description


**`Result` object**
- *methods*
- get_runs()
- get_run(number) where number is an int starting from 1
- get_units()
- get_names()
- get_descriptions()
- get_column(name) where name is a string and case insensitive.
    - Almost similar to get_runs() but only give a particular column.

Functions
---------

`NIR_chart_month`
- net irrigation requirement chart by month
- wet, normal and dry corresponds to 80%, 50%, 20% probability of exceedance respectively.

`NIR_chart_decade`
- net irrigation requirement chart by decade (10-day)
- wet, normal and dry corresponds to 80%, 50%, 20% probability of exceedance respectively.

`boxplot`
- boxplots of a variable from list of results

**`tsplot`**
- time series plot of a variable
- derived from the multiple runs
- uses the average as the main plot
- shows 
