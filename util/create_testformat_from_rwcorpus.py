import pandas as pd
import os
import argparse


def create_testformat(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_files = [x for x in os.listdir(input_dir) if x[-4:] == ".tsv"]
    for file in input_files:
        data = pd.read_csv(os.path.join(input_dir, file), sep="\t", quoting=3, encoding="utf-8", na_values=[])
        if "stwr" not in data.columns:
            print("file {} is missing 'stwr' column".format(file))
            exit(0)

        stwr_annos = list(data["stwr"])
        rw_type_dict = {"direct": [], "indirect": [], "reported":[], "freeIndirect":[]}
        for anno in stwr_annos:
            anno_fields = [x.split(".")[0] for x in anno.split("|")]
            for rwtype in rw_type_dict.keys():
                if rwtype in anno_fields:
                    rw_type_dict[rwtype].append(rwtype)
                else:
                    rw_type_dict[rwtype].append("x")
        for rwtype in rw_type_dict.keys():
            data[rwtype] = rw_type_dict[rwtype]

        data.to_csv(os.path.join(output_dir, file), sep="\t", index=False, encoding="utf-8")


# call the starting method
if __name__ == "__main__":
    help_text = """
    Create test files from files from Corpus REDEWIEDERGABE
    
    Takes a directory with files from Corpus REDEWIEDERGABE in column-based text format (tsv) and
    transforms the annotation column in such a way that the resulting files can be used as input to
    rwtagger.py in test mode.
    Output files are written to the output directory.
    """
    parser = argparse.ArgumentParser(description=help_text, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input_dir", help="input directory")
    parser.add_argument("output_dir", help="output directory")
    args = parser.parse_args()
    create_testformat(args.input_dir, args.output_dir)