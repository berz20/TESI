import os
for file in os.listdir(ODMR):
    if file.endswith("ch0_range0.dat"):
        print(os.path.join(ODMR, file))
        s = '"'
        f = ODMR+"/"+file
        splitting(f)


