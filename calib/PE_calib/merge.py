import tables
import h5py
import argparse

def readbyh5(filename):
    keys = []
    for i in filename:
        print(i)
        #h = h5py.File(i,'r')
        print(h.keys())
        for j in h.keys():
            keys.append(j)
            print(type(h[j][:].dtype))
            for k in h[j][:].dtype[0]:
                print(k[0])

                
def readbytables(filename):
    keys = []
    for i in filename:
        print(i)
        h = tables.open_file(i)
        print(h.root.keys())
        
parser = argparse.ArgumentParser(description='Process some integers.')   
parser.add_argument('--filename', metavar='N', type=str, nargs='+',
                    help='merge h5')

args = parser.parse_args()
print(args.filename)
readbytables(args.filename)
