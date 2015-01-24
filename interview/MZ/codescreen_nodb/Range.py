import getpass
import logging

''' Design Ideas
    I use list in this module because I am not clear how big the data set will be.
    List is one of the simplest data structure in python, 
    however, if the data is huge, I might try to use matrix (numpy) instead of list.
    Under the circumstances without knowing the real purpose, I decide not to over-engineering.
    My basic assumption is: all ranges are given in integers.
    Because range() function is used, this RangeModule will fail if the range is not given in integers.
'''


class RangeModule:

    def __init__(self, **kwargs):
        logging.debug('Initializing RangeModule class.')
        self.saved_range = []

    def QueryRange(self, lower, upper):
        self.MakeRangeIncludive(upper)
        new_range = range(lower, upper)
        exclusive_range = set(new_range) - set(self.saved_range)
        #exclusive_range = [i for i in new_range if i not in self.saved_range]
        if len(exclusive_range) > 0:
            logging.debug('New range found %i - %i' %(lower, upper))
            return False, exclusive_range
        else:
            logging.debug('The range %i - %i is found in the database.' %(lower, upper))
            return True, exclusive_range

    def RemoveRange(self, lower, upper):
        self.MakeRangeIncludive(upper)
        self.saved_range = list(set(self.saved_range) - set(range(lower, upper)))
        logging.debug('Removing %i - %i from the existing database.' %(lower, upper))

    def AddRange(self, lower, upper):
        self.MakeRangeIncludive(upper)
        included, exclusive_range = self.QueryRange(lower, upper)
        if not included:
            logging.debug('Adding %i - %i to the existing database.' %(lower, upper))
            self.saved_range.extend(exclusive_range)
            self.saved_range.sort()

    def MakeRangeIncludive(self, upper):
        return upper+1

    def PrintRange(self):
        self.saved_range.sort()
        logging.info('min = %i , max = %i' %(self.saved_range[0], self.saved_range[-1]))
