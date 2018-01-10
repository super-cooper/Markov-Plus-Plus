import sys
from typing import List
from mpp import MarkovModel
import getopt

options = 'd:hn:'
long_options = ['dirs', 'help', 'num-chars=']
HELP_TEXT = 'python3 markovgen.py -d "<directory regex pattern>" -n <num-chars>'


def main(argv: List[str]) -> None:
    """Main function"""
    dir_pattern = None
    n = 1000

    try:
        opts, args = getopt.getopt(argv, options, long_options)
    except getopt.GetoptError:
        print(HELP_TEXT)
        sys.exit(1)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(HELP_TEXT)
            sys.exit(0)
        elif opt in ('-d', 'dirs'):
            dir_pattern = arg
        elif opt in ('n', 'num-chars='):
            n = int(arg)
        else:
            print('Argument {}: {} not recognized'.format(opt, arg))
            sys.exit(1)

    model = MarkovModel(dir_pattern)

    if dir_pattern is None or len(model) == 0:
        print('Must supply directory with working files')
        sys.exit(1)

    out = model.get_start()
    prev = out

    while len(out) < n:
        c = model.random_suffix(prev)
        prev = prev[1:] + c
        out += c

    print(out)


if __name__ == '__main__':
    main(sys.argv[1:])
