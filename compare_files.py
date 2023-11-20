import argparse

# define input args
parser = argparse.ArgumentParser(
    prog='compare_files.py',
    description='Compare two text files for equality')

parser.add_argument('file1', type=str)
parser.add_argument('file2', type=str)
args = parser.parse_args()

f1 = args.file1
f2 = args.file2

eps = 1e-6


def print_results(current, expected):
    print('Expected:\t\tCurrent:')
    for s1, s2 in zip(expected, current):
        print(f'{s1.strip()}\t{s2.strip()}')


def check_results():
    with open(f1, 'r') as curr, open(f2, 'r') as expect:
        # compare strings (potentially with floating point check)
        for s1, s2 in zip(curr, expect):
            s1 = s1.strip()
            s2 = s2.strip()
            if s1 != s2:
                try:
                    if abs(float(s1) - float(s2)) > eps:
                        print(f'Results do not match.')
                        print(f'Expected "{s2.strip()}" but got "{s1.strip()}"')
                        print('\nFull comparison:')
                        print_results(expect, curr)
                        return False
                except ValueError:
                    print(f'Results do not match.')
                    print(f'Expected "{s2.strip()}" but got "{s1.strip()}"')
                    print('\nFull comparison:')
                    print_results(expect, curr)
                    return False

        return True


if not check_results():
    raise ValueError("File contents do not match")

print("Success: File contents are identical!")
