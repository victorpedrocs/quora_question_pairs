import sys, getopt
import stratified_sample as ss

def main(argv):
    size = ''
    out_type = ''
    try:
        opts, args = getopt.getopt(argv, "hs:")
    except getopt.GetoptError:
        print("-s <sample_size>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("-s <sample_size>")
            sys.exit()
        elif opt == '-s':
            sample_size = arg
        elif opt == '-t':
            out_type = arg

    ss.generate(int(size), out_type)



if __name__ == "__main__":
   main(sys.argv[1:])
