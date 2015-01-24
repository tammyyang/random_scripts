#!/usr/bin/env python
#-------------------------------------------------------------------------------------------------
# Filename: convert_eps.py
# Description: convert eps files to other file type
# Created: 09-May-2008 Wan-Ching Yang
#-------------------------------------------------------------------------------------------------
import os
import sys
from glob import glob
from time   import *
from string import *
from getopt     import getopt, GetoptError
import commands

#-------------------------------------------------------------------------------------------------
ANS={"y":["y","Y","yes","Yes","YES","T","True","true"],"n":["n","N","No","no","NO","F","False","false"]}
#-------------------------------------------------------------------------------------------------

def usage():
    print \
          '''
  Usage:
    convert_eps.py [option]
      option
      -h print usage
      --type= output type of the graphs (default: "jpg")
      --dir= directory to search for eps files (default: ".")
      --keyw= keyword of the graphs (default: "*" )
      --suffix= suffix of the file name (default: "")
                     '''
#-------------------------------------------------------------------------------------------------

def decodeCommandLine():
  try:
    opts, args = getopt(sys.argv[1:], "htdks:", ["help", "dir=", "type=", "keyw=", "suffix="])
  except GetoptError, err:
    print str(err)
    usage()
    sys.exit(2)
  type="jpg"
  dir="./"
  keyw="*"
  suffix=""
  rmeps=False
  for o, a in opts:
    if o in ("-h", "--help"):
      usage()
      sys.exit()
    elif o in ("-t", "--type"):
      type=a
    elif o in ("-d", "--dir"):
      dir="./"+a
    elif o in ("-s", "--suffix"):
      suffix="_"+a
    elif o in ("-k", "--keyw"):
      keyw=a
  return type, dir, keyw, suffix

#-------------------------------------------------------------------------------------------------

def main():
  type, dir, keyw, suffix= decodeCommandLine()
  brmeps=False
  if not os.path.exists("./EPSFigs"):
    os.system("mkdir ./EPSFigs") 
  if not os.system("ls %s/*%s*.eps" %(dir, keyw)):
    epsset=commands.getoutput("ls %s/*%s*.eps" %(dir, keyw))

    for epsfile in epsset.split("\n"):
      newfile=os.path.splitext(epsfile)[0]+suffix+"."+type
      print "Converting %s to %s, and %s to ./EPSFigs" %(epsfile, newfile, epsfile)
      os.system("convert %s %s" %(epsfile, newfile))
      os.system("mv %s ./EPSFigs" %epsfile)
  else:
    print "No eps files found"
    sys.exit(2)

#-------------------------------------------------------------------------------------------------
main()

