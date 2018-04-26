from optparse import OptionParser

if __name__ == "__main__":
    import sys
    op = OptionParser(usage='Usage: %prog [options]')
    op.add_option("--input",
              dest="fname",
              help="Input file to retag")
    op.add_option("--output",
              dest="oname",
              help="Output file where to place the result")
    op.add_option("--target_column",
              dest="targetcol",
              help="Target column to concat")

    (opts, args) = op.parse_args()

    if not opts.fname or not opts.oname:
        print("Needed --input --output parameters")
        sys.exit()
    with open(opts.fname) as f:
        content = f.readlines()
    targetcol = int(opts.targetcol)
    with open(opts.oname, "w+") as f:
        for line in content:
            items = line.split("\t")
            if len(items) < 2:
                f.write('\n\n')
                continue
            new_line = items[targetcol] + " "
            f.write(new_line)
