#!/usr/bin/env python
import subprocess

out_bytes = subprocess.check_output(['ps', '-aux'])
out_text = out_bytes.decode('utf-8')
out_text = out_text.strip().split('\n')
for line in out_text:
    if 'tf_main.py' in line:
        print line
        id = line.strip().split()[1]
        subprocess.call('kill %s' % id, shell=True)
