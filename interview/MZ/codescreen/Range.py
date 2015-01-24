import getpass
import logging
import sys
import MySQLdb

''' Design Ideas
    I use string and list in this module because I am not clear how big the data set will be.
    The advantage of using string is that I can control the size of the database (since how big the input is unknown).
    The disadvantage is line 60, in QueryRange can cost a lot of memory.
    List is one of the simplest data structure in python, 
    however, if the data is huge, I might try to use matrix (numpy) instead of list.
    Under the circumstances without knowing the real purpose, I decide not to over-engineering.
    My basic assumption is: all ranges are given in integers.
    Because range() function is used, this RangeModule will fail if the range is not given in integers.

    The RangeModule requires an existing database to query.
    The default database is testdb with only one user (root)
'''

class RangeModule:

    def __init__(self, **kwargs):
        logging.debug('Initializing RangeModule class.')
        self.input_parameters = {'username': 'root',
                                 'password': None,
                                 'tablename': 'RANGESTABLE',
                                 'column': 'RANGESTRING',
                                 'database': 'testdb'}
        for key in [key for key in kwargs.keys() if kwargs[key] != None]:
            self.input_parameters[key] = kwargs[key]
        self.db = self.ConnectDB(self.input_parameters['username'], self.input_parameters['password'], self.input_parameters['database'])
        self.cursor = self.db.cursor()
        self.CreateNewRangeTable()

    def GetCurrentRange(self):
        self.cursor.execute('SELECT * from %s' %(self.input_parameters['tablename']))
        results = self.cursor.fetchall()
        if len(results) > 1:
            logging.error('More than one data is in %s' %self.input_parameters['tablename'])
            return
        elif len(results) == 0:
            logging.debug('No existing data found')
            sql = "INSERT INTO %s(%s) VALUES ('')" %(self.input_parameters['tablename'], self.input_parameters['column'])
            self.cursor.execute(sql)
            self.db.commit()
            return []
        return results[0][0].split("-")

    def CreateNewRangeTable(self):
        if not self.cursor.execute("SHOW TABLES LIKE '%s'" %self.input_parameters['tablename']):
            logging.debug('No existing table found, create a new table.')
            sql = 'CREATE TABLE %s (%s  MEDIUMTEXT NOT NULL)' %(self.input_parameters['tablename'], self.input_parameters['column'])
            self.cursor.execute(sql)

    def ConnectDB(self, username, password, database):
        if password == None:
            password = getpass.getpass('Please input the password for user %s' %username)
        return MySQLdb.connect('localhost', username, password, database )

    def QueryRange(self, lower, upper):
        self.MakeRangeIncludive(upper)
        current_range = self.GetCurrentRange()
        new_range = map(str, range(lower, upper))
        logging.debug('size of new_range: %i' %sys.getsizeof(new_range))
        exclusive_range = set(new_range) - set(current_range)
        if len(exclusive_range) > 0:
            logging.debug('New range found %i - %i' %(lower, upper))
            return False, exclusive_range
        else:
            logging.debug('The range %i - %i is found in the database.' %(lower, upper))
            return True, exclusive_range

    def RemoveRange(self, lower, upper):
        self.MakeRangeIncludive(upper)
        current_range = self.GetCurrentRange()
        new_range = set(current_range) - set(map(str, range(lower, upper)))
        logging.debug('Removing %i - %i from the existing database.' %(lower, upper))
        self.UpdateRange(new_range)

    def UpdateRange(self, range_list):
        s = '-'.join(range_list)
        sql = "UPDATE %s SET %s = '%s'" % (self.input_parameters['tablename'], self.input_parameters['column'], s)
        logging.debug('Updating database...')
        self.cursor.execute(sql)
        self.db.commit()

    def AddRange(self, lower, upper):
        self.MakeRangeIncludive(upper)
        current_range = self.GetCurrentRange()
        included, exclusive_range = self.QueryRange(lower, upper)
        if not included:
            logging.debug('Adding %i - %i to the existing database.' %(lower, upper))
            current_range.extend(exclusive_range)
            current_range.sort()
            self.UpdateRange(current_range)

    def Finish(self):
        self.db.close()

    def MakeRangeIncludive(self, upper):
        return upper+1
