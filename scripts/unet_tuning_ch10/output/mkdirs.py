import os

for i in range(24):
    try:
        os.mkdir(str(i))
    except:
        pass