# System
import argparse

def restricted_F(x):
    y=float(x)
    if y <= 0.0 or y >= 1.0:
        raise argparse.ArgumentTypeError("%r not in range (0.0, 1.0)"%(y,))
    return y

def restricted_Omega(x):
    y=float(x)
    if y <= -1.0 or y >= 1.0:
        raise argparse.ArgumentTypeError("%r not in range (-1.0, 1.0)"%(y,))
    return y

def positive_float(x):
    y=float(x)
    if y <= 0.0:
        raise argparse.ArgumentTypeError("%r not positive"%(y,))
    return y
