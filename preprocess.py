import sys, getopt
import quora_dataset_builder as dsb

def main(argv):
    out_type = ''
    sample_size = ''
    try:
        opts, args = getopt.getopt(argv, "ht:")
    except getopt.GetoptError:
        print("-t [stemm, tokenize, simple]")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("-t [stemm, tokenize, simple]")
            sys.exit()
        elif opt == '-t':
            out_type = arg

    dsb.generate(out_type)



if __name__ == "__main__":
   main(sys.argv[1:])
