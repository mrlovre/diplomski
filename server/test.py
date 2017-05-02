from utility.functions import *
import numpy as np
import bdateutil as bu
from datetime import date


def main():
    print(bu.isbday(date(2017, 1, 2)))


if __name__ == '__main__':
    main()
