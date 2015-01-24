#!/usr/bin/env python
from Range import RangeModule
import logging

''' This is a testing script to verify if RangeModule works WITHOUT using any database 
    It takes 8s to run this testing script on my machine which query a range as large as ten million.
    (Genuine Intel(R) CPU U7300  @ 1.30GHz with 4G RAM)

    time(python main.py)
    2014-05-24 12:56:26,304 DEBUG Initializing RangeModule class.
    2014-05-24 12:56:26,305 DEBUG New range found 200 - 5000
    2014-05-24 12:56:26,305 DEBUG New range found 0 - 2
    2014-05-24 12:56:26,306 DEBUG Adding 0 - 2 to the existing database.
    2014-05-24 12:56:29,749 DEBUG New range found 3000 - 11123980
    2014-05-24 12:56:29,870 DEBUG Adding 3000 - 11123980 to the existing database.
    2014-05-24 12:56:33,249 DEBUG Removing 200 - 300 from the existing database.
    2014-05-24 12:56:34,023 INFO min = 0 , max = 11123979

    real    0m8.567s
    user    0m7.512s
    sys 0m0.956s

'''

logging.basicConfig(level=True, format='%(asctime)s %(levelname)s %(message)s')
r = RangeModule()
r.QueryRange(200, 5000)
r.AddRange(0,2)
r.AddRange(3000, 11123980)
r.AddRange(99000, 1223980)
r.RemoveRange(200,300)
r.PrintRange()
