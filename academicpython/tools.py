"""
General purpose tools
"""
import sys, os
import subprocess
import logging
from time import time

logging.basicConfig(level=logging.WARNING)

def makedir(*argv):
    """ make a directory if INTERP does not exist """
    for path in argv:
        os.system('mkdir -p {}'.format(path))

def betterRun(cmd, prt=True, check=True):

    process = subprocess.run(cmd, shell=1, check=check, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,  universal_newlines=True)
    if prt:
        print(process.stdout)
    return process.stdout, process.stderr

def getarg(arg, value=None):
    """Parse command line inputs and return True or False.

    Usage:
        if getarg("-p"):
            print("debug print")
        if getarg("-o", "PDF"):
            fmt = "PDF"
    """

    if len(sys.argv) <= 2:
        return False
    if arg in sys.argv[1:]:
        if value is None:
            return True
        else:
            idx = sys.argv.index(arg)
            try:
                return sys.argv[idx + 1] == value
            except IndexError:
                return False
    return False

def getlog(name, level='warning'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.__dict__[level.upper()])
    return logger

class mylogging:

    level_dict = {'debug': 0, 'info': 1, 'warning': 2, 'error': 3, 'critical': 4}

    def __init__(self,
                 level='info',
                 fmt='MYLOGGING: {level}: {name}: {msg}',
                 name=__name__):
        self.level = self.level_dict[level]
        self.levelstr = level
        self.fmt = fmt
        self.name = name

    def set_fmt(self, fmt):
        self.fmt = fmt

    def setLevel(self, level):
        self.level = self.level_dict[level]
        self.levelstr = level

    def debug(self, msg):
        if self.level <= 0:
            print(self.fmt.format(level='debug', name=self.name, msg=msg))

    def info(self, msg):
        if self.level <= 1:
            print(self.fmt.format(level='info', name=self.name, msg=msg))

    def warning(self, msg):
        if self.level <= 2:
            print(self.fmt.format(level='warning', name=self.name, msg=msg))

    def error(self, msg):
        if self.level <= 3:
            print(self.fmt.format(level='error', name=self.name, msg=msg))

    def critical(self, msg):
        if self.level <= 4:
            print(self.fmt.format(level='critical', name=self.name, msg=msg))

def archiveRun():
    """Run a jobs in the following steps:
        1. If intermediate date exists, use that data
        2. If intermediate date does not exist, create that data tha use that data
    """
    return

class timeit():

    def __init__(self, ):
        self.t1 = time()

    def a(self):
        T1 = time()
        self.t1 = time()

    def b(self):
        dt = time() - self.t1
        print(f"\nTime elapsed: {dt:.1f} seconds\n")

    def start(self):
        self.a()

    def end(self):
        self.b()
