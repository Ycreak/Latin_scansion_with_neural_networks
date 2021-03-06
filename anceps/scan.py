import argparse
import datetime
import json
import sys
from tqdm import tqdm
import warnings
from progress.bar import Bar

from anceps.analyze import analyse
from anceps.meter import Meter
from anceps.verse import Verse
from anceps.word import Word


class Scan():

    def __init__(self, hexameter_candidates):
        # parse command line arguments:
        p = argparse.ArgumentParser(description="Scan a text")
        p.add_argument("-input", type=str, default='./anceps/test.txt',
                    help="input file with one verse of poetry per line. Optionally, the verse can be "
                            "preceded by a unique index and a tab (set -input_index to True in this case)")
        p.add_argument("-output", type=argparse.FileType("w"), default='./anceps/test.json',
                    help="output file name (should end with .json)")
        p.add_argument("-meter", type=str, choices=list(Meter.METERS.keys()), default='hexameter',
                    help="meter to scan the text with")
        p.add_argument("-manual_file", type=str, default=None,
                    help="file from which to read manual scansions and to which to write lines that "
                            "require manual scansions")
        p.add_argument("-dictionary", type=str, default=None,
                    help="MQDQ dictionary file to use during scansion")
        p.add_argument("-ac", type=int, default=3,
                    help="If there is a way to scan a word which can only be find in MqDq dictionary "
                            "(such as pa_tri*s), this parameter specifies the number of authors the "
                            "scansion has to appear in for it to be considered valid by the program")
        p.add_argument("-tc", type=int, default=5,
                    help="If there is a way to scan a word which can only be find in MqDq dictionary "
                            "(such as pa_tri*s), this parameter specifies the number of times this "
                            "scansion has to appear in the corpus for it to be considered valid by the "
                            "program")
        p.add_argument("-cutoff", type=float, default=0.05,
                    help="If there are two ways to scan a line and one way has this probability or lower"
                            ", the frequent scansion will be selected automatically without "
                            "consulting the user")
        p.add_argument("--precise", dest="precise", action="store_true",
                    help="require the quantity of every syllable to be determined. This will, for "
                            "instance force the program to differentiate brevis in longo from longum at "
                            "the end of the line")
        p.add_argument("--interactive", dest="interactive", action="store_true",
                    help="prompt the user to select the correct scansion when the program has "
                            "several alternatives")
        p.add_argument("--input_index", dest="input_index", action="store_true",
                    help="assume that the input file contains verse indices (see help message for "
                            "input parameter) and use these indices for output")
        p.add_argument("--anceps", action="store_true",
                    help="dummy")        
        p.add_argument("--pedecerto", action="store_true",help="dummy")          
        p.add_argument("--candidates", action="store_true",help="dummy")          

        p.add_argument("--no_diphthongs", dest="diphthongs", action="store_false",
                    help="use if the input text has 'e' instead of 'ae' and 'oe' (this happens in "
                            "some Renaissance texts)")
        p.add_argument("--add_failed_to_manual", dest="add_failed", action="store_true",
                    help="if True, the lines that the program failed to scan will be added to the "
                            "manual file so that they can later be revisited")
        p.set_defaults(precise=False, input_index=False, interactive=False, diphthongs=True,
                    add_failed=False)
        args = p.parse_args(sys.argv[1:])
        if args.interactive and not args.manual_file:
            warnings.warn("Scansions chosen in interactive mode are lost, if manual_file is not specified.")

        # load manually scanned lines into Verse dictionary:
        # if args.manual_file:
        #     Verse.read_manual_file(args.manual_file)

        Word.DIPHTHONGS = True
        Word.AUTHOR_COUNT = 3
        Word.TOTAL_COUNT = 5
        Word.load_mqdq_dict('./anceps/MqDqMacrons.json')
        Word.load_morpheus_dict("./anceps/MorpheusMacrons.txt")
        
        # Word.load_mqdq_dict(None)
        # Word.load_morpheus_dict(None)
        
        # Word.load_morpheus_dict("../../data/MorpheusMacrons.txt")
        Verse.CUTOFF = 0.05

        # args.input = './anceps/test.txt'
        # args.output = './anceps/test.json'
        # print('hello')

        args.meter = Meter.METERS[args.meter]
        if not isinstance(args.meter, tuple):  # tuples are used for meters like elegiacs
            args.meter = (args.meter, )
        # f = open(args.input, "r")
        # lines = f.readlines()

        # We give as input the lines given to this class from prose.py
        lines = hexameter_candidates

        data = {"text": {}}

        # print("Scansion in progress...")

        # with Bar('Anceps: creating candidate list', max=len(lines)) as bar:
        for i, line in enumerate(lines):
            # if args.input_index:
            #     key, verse = line.rstrip("\n").split("\t")
            # else:
            key, verse = str(i), line
            
            data["text"][key] = {"verse": verse}
            verse = Verse(verse)
            curr_meter = args.meter[i % len(args.meter)]
            scansion = verse.scan(curr_meter, args.precise, args.interactive, args.add_failed)
            if scansion:
                data["text"][key]["scansion"] = str(scansion)
                data["text"][key]["pattern"] = \
                    str(curr_meter.get_matching_scansions(scansion, args.precise)[0])
            else:
                data["text"][key]["scansion"], data["text"][key]["pattern"] = "", ""
            data["text"][key]["method"] = verse.scansion_method
            data["text"][key]["flags"] = verse.flags
            data["text"][key]["meter"] = curr_meter.name

                # bar.next()

        # run analysis on the scanned text
        data["stats"] = analyse(data["text"])
        now = datetime.datetime.now()
        data["createdOn"] = {"day": now.day, "month": now.month, "year": now.year}

        # the json file where the output must be stored
        # out_file = open(args.output, "w")
        # json.dump(data, out_file, indent = 2)
        # out_file.close()

        # Lastly. save the result to a variable to be read by prose.py
        self.result = data

        # json.dump(data, args.output, indent=2)
        if args.manual_file:
            Verse.save_manual_file(args.manual_file)
