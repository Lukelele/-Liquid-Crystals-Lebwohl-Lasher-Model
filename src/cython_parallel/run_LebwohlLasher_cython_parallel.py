import sys
from LebwohlLasher_cython_parallel import main

if int(len(sys.argv)) == 5:
    main(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
else:
    print("Usage: {} ".format(sys.argv[0]))

