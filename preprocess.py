#!/Users/vpedro/PESC/2017.1/data_mining/assignments/quora/bin/python
import sys, getopt
import dataset_builder as dsb

def print_help():
    print("\t-s:\t\tSingle Question per line dataset")
    print("\t-c:\t\tConcat question in one line")

def main(argv):
    out_type = ''
    sample_size = ''
    try:
        opts, args = getopt.getopt(argv, "hsc")
    except getopt.GetoptError:
        print_help()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print_help()
            sys.exit()
        elif opt == '-s':
            dsb.generate_one_question_per_line()
        elif opt == '-c':
            dsb.generate_concat()



if __name__ == "__main__":
   main(sys.argv[1:])
