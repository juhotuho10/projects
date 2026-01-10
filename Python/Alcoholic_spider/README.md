# Alcoholic Spider
A fun web crawler suggested by a friend I made back in in September 2021

The program fetches the HTML code from the Alko website and then extracts product data for all alcoholic beverages. From that it calculates the alcohol percentage per liter per euro value and determines the best cost-to-alcohol ratio drink from Alko.

couple of notes for running it:
- The program NEEDS firefox to be installed, because we use the default user firefox profile with the webdrive to access Alko website
- Sadly currently windows only
- The program has to parse a lot of HTML code, so loading it into RAM and running it can be a bit slow, taking about 10-30 minutes
