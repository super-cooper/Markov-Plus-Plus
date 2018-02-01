#!/bin/python3.6

import json
import sys

from mpp import Utils

args = sys.argv[1:]
if len(args) < 2:
    print('Usage: extract_jokes.py <json-dir-glob> <out-dir>')

dirs = args[0]
out = args[1]

for file in Utils.get_files(dirs):
    raw = []
    if not file.endswith('.json'):
        print(file, 'is not a json file, did not read')
        continue

    print('Reading', file)
    with open(file, 'r', encoding='utf-8') as f:
        jokes = json.loads(f.read())

    # Each joke is represented as a dict of attributes
    for joke in jokes:
        full = ''
        if 'title' in joke:
            title = joke['title'].strip()
            # Trying to preserve the full context of the Reddit jokes (some will be missed unfortunately)
            if title.endswith('?'):
                full += title + ' '
            else:
                continue
        full += joke['body']
        raw.append(full)

    outfile = out + file[file.rfind('/'):-5] + '.txt'
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write(Utils.INSTANCE_DELIMITER.join(raw))
    print('Jokes extracted.')


