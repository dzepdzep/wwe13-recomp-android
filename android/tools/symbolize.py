import sys, re, subprocess
lib = sys.argv[1]; src = sys.argv[2]; filt = sys.argv[3] if len(sys.argv) > 3 else None
for line in open(src, errors='replace'):
    line = line.strip()
    if filt and not re.search(filt, line): continue
    head = line.split(' libmain')[0].split(' ?+')[0]
    print('==', head)
    addrs = re.findall(r'libmain\.so\+(0x[0-9a-f]+)', line)
    if not addrs: continue
    out = subprocess.run(['llvm-symbolizer-18', '--obj=' + lib] +
                         [hex(int(a, 16) - 1) for a in addrs], capture_output=True, text=True).stdout.split('\n\n')
    for a, o in zip(addrs, out):
        fn = o.strip().split('\n')[0][:150]
        print('   ', a, fn)
