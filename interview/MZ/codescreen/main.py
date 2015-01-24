#!/usr/bin/env python
from Range import RangeModule
import logging

''' This is a testing script to verify if RangeModule works 
    It takes 6s to run this testing script on my machine which query a range as large as one million.
    (Genuine Intel(R) CPU U7300  @ 1.30GHz with 4G RAM)

    time(python main.py)
    2014-05-24 12:21:23,211 DEBUG Initializing RangeModule class.
    Please input the password for user root
    2014-05-24 12:21:24,381 DEBUG No existing table found, create a new table.
    ()
    2014-05-24 12:21:24,508 DEBUG No existing data found
    2014-05-24 12:21:24,577 DEBUG New range found 200 - 5000
    2014-05-24 12:21:24,578 DEBUG New range found 0 - 2
    2014-05-24 12:21:24,578 DEBUG Adding 0 - 2 to the existing database.
    2014-05-24 12:21:24,578 DEBUG Updating database...
    2014-05-24 12:21:25,340 DEBUG New range found 3000 - 1123980
    2014-05-24 12:21:25,362 DEBUG Adding 3000 - 1123980 to the existing database.
    2014-05-24 12:21:25,623 DEBUG Updating database...
    2014-05-24 12:21:27,423 DEBUG Removing 200 - 300 from the existing database.
    2014-05-24 12:21:27,734 DEBUG Updating database...

    real    0m5.901s
    user    0m2.188s
    sys 0m0.296s

'''

logging.basicConfig(level=True, format='%(asctime)s %(levelname)s %(message)s')
r = RangeModule()
r.QueryRange(200, 5000)
r.AddRange(0,2)
r.AddRange(3000, 1123980)
r.AddRange(500, 92345)
r.RemoveRange(200,300)
r.Finish()
