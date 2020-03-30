import pandas as pd
import os
import argparse
import logging

def tsv_to_excel(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_files = [x for x in os.listdir(input_dir) if x[-4:] == ".tsv"]
    for file in input_files:
        data = pd.read_csv(os.path.join(input_dir, file), sep="\t", quoting=3, encoding="utf-8", na_values=[])
        writer = pd.ExcelWriter(os.path.join(output_dir, file.replace(".tsv", ".xlsx")))
        data.to_excel(writer, index=None)
        writer.save()


# call the starting method
if __name__ == "__main__":
    help_text = """
    Converts tsv files in the input directory to Excel files (.xlsx)
    and writes them to the output directory.
    """
    parser = argparse.ArgumentParser(description=help_text, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input_dir", help="input directory")
    parser.add_argument("output_dir", help="output directory")
    args = parser.parse_args()
    tsv_to_excel(args.input_dir, args.output_dir)