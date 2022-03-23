# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from datetime import datetime
import os

def print_hi(name):
    f = open("/home/pi/code/LicencePlatesDetector/logs.txt", "a")
    now = datetime.now()
    f.write(f"\nLOGGING {now}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
