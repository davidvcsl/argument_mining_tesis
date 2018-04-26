from optparse import OptionParser

def simplifyTag(tag, deep_clean):
    if tag == "O":
        return tag
    tag = tag.strip("B-").strip("I-")
    for num in [1,2,3,4,5,6,7,8,9,0]:
        tag = tag.replace(str(num), "")
    tag = tag.replace("::", ":")
    if deep_clean:
        tag = tag.split(":")[0]
    if not tag.endswith("\n"):
        tag = tag + "\n"
    return tag

if __name__ == "__main__":
    import sys
    op = OptionParser(usage='Usage: %prog [options]')
    op.add_option("--input",
              dest="fname",
              help="Input file to retag")
    op.add_option("--output",
              dest="oname",
              help="Output file where to place the result")
    op.add_option("--deepclean",
              dest="deepclean",
              default=1,
              help="Clean only one-word tag")

    (opts, args) = op.parse_args()

    if not opts.fname or not opts.oname:
        print("Needed --input --output parameters")
        sys.exit()
    with open(opts.fname) as f:
        content = f.readlines()
    deep_clean = opts.deepclean == 1
    with open(opts.oname, "w+") as f:
        for line in content:
            items = line.split("\t")
            if len(items) < 2:
                f.write(line)
                continue
            items[2] = simplifyTag(items[2], deep_clean)
            new_line = "\t".join(items)
            f.write(new_line)
