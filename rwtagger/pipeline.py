from typing import Dict

import tagger
import os
import shutil
import logging


class Pipeline:
    def __init__(self, use_gpu, log_level):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.use_gpu = use_gpu
        self.log_level = log_level

    def predict(self, inputdir, outputdir, rwtype_list, input_format, chunk_len, test_scores, confidence_vals, config_path):
        """
        Calls the recognizers for each STWR type in turn; results are stored in output dir;
        temporary results are stored in directory "temp" which is created at the script location

        predicted format looks like this:
        tok sentstart   direct  indirect    reported free_indirect

        :param inputdir: directory with input files (tsv or txt)
        :param outputdir: directory for output files (will be overwritten with new results)
        :param rwtype: one of: direct, indirect, reported, free_indirect
        :return:
        """
        # check whether there are files of the correct input type in the input dir before doing anything
        input_files = [x for x in os.listdir(inputdir) if x[-4:] == "." + input_format]
        if len(input_files) == 0:
            self.logger.info("Input directory ({}) does not contain any files of type '{}'. Tagging aborted.".format(inputdir, input_format))
            exit(0)

        # check for each rwtype whether the model exists, before starting the pipeline
        # start by determining the paths via config file (if the config file is not found, default paths are used)
        model_paths_dict = self.read_config(config_path)
        self.logger.info("Current model paths: {}".format(model_paths_dict))

        available_rwtypes = []
        curr_path = os.path.dirname(os.path.abspath(__file__))
        for rw_type in rwtype_list:
            model_path = os.path.join(curr_path, "models", model_paths_dict[rw_type], "final-model.pt")
            if not os.path.exists(model_path):
                logging.warning(
                    "Predicting {} aborted. Model not found at path '{}'. Please download a model and put it into "
                    "the appropriate directory. The model file must be named final-model.pt.".format(rw_type,
                                                                                                     model_path))
            else:
                available_rwtypes.append(rw_type)

        if len(available_rwtypes) == 0:
            exit(0)

        # for the first available rwtype:
        # initialize the tagger for this STWR type
        self.logger.info("Start predicting {}  (inputdir: {}, outputdir: {})".format(available_rwtypes[0], inputdir, outputdir))
        # if it already exists, remove the outputdir and create an new one
        if os.path.exists(outputdir):
            try:
                shutil.rmtree(outputdir)
            except PermissionError:
                self.logger.error("Could not remove old output directory: {}.\nPlease remove manually. Tagging aborted.".format(outputdir))
                exit(0)
        os.makedirs(outputdir)
        rwtagger = tagger.RWTagger(self.use_gpu, self.log_level)
        rwtagger.predict(inputdir, outputdir, available_rwtypes[0], input_format=input_format, chunk_len=chunk_len,
                         test_scores=test_scores,
                         output_confidence=confidence_vals,
                         special_model_path=model_paths_dict[available_rwtypes[0]])
        self.logger.info("Finished predicting {}".format(available_rwtypes[0]))
        # for all following rwtypes: input is always tsv and tempdir is used as inputdir
        # however, if tempdir is empty (can happen if the tagging was aborted) use inputdir instead
        curr_path = os.path.dirname(os.path.abspath(__file__))
        tempdir = os.path.join(curr_path, "temp")
        self.logger.info("Creating tempdir: {}".format(tempdir))
        if len(available_rwtypes) > 1:
            # remove any old tempdirs and create an new one
            if os.path.exists(tempdir):
                try:
                    shutil.rmtree(tempdir)
                except PermissionError:
                    self.logger.error("Could not remove old temp directory: {}.\nPlease remove manually. Tagging aborted.".format(tempdir))
                    exit(0)
            os.makedirs(tempdir)
            for rwtype in available_rwtypes[1:]:
                # move all the tsv files from outputdir to temp dir
                tsv_files = [x for x in os.listdir(outputdir) if x[-4:] == ".tsv"]
                for file in tsv_files:
                    shutil.move(os.path.join(outputdir, file), os.path.join(tempdir, file))
                self.logger.info("Start predicting {} (inputdir: {}, outputdir: {})".format(rwtype, tempdir, outputdir))
                # initialize the tagger for this STWR type
                rwtagger = tagger.RWTagger(self.use_gpu, self.log_level)
                # inputtype in always "tsv" now and input_dir is the temp dir
                rwtagger.predict(tempdir, outputdir, rwtype, input_format="tsv", chunk_len=chunk_len, test_scores=test_scores,
                                 output_confidence=confidence_vals, special_model_path=model_paths_dict[rwtype])
                self.logger.info("Finished predicting {}".format(rwtype))
            # when all types have been predicted, remove the temp dir
            try:
                shutil.rmtree(tempdir)
            except PermissionError:
                self.logger.warning("Predictions finished, but could not remove temp directory: {}.".format(tempdir))

    def read_config(self, configpath):
        #initialize dict with default paths
        rw_path_dict: Dict[str, str] = {"direct":"direct", "freeIndirect":"freeIndirect", "indirect":"indirect", "reported":"reported"}
        # try reading config
        if not os.path.exists(configpath):
            self.logger.warning("Config file not found at {}. Using default model paths for STWR types.".format(configpath))
        else:
            with open(configpath, "r", encoding="utf8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line != "" and (not line.startswith("#")):
                        fields = line.split("@")
                        print(fields)
                        if fields[0] in ["direct", "freeIndirect", "indirect", "reported"]:
                            rw_path_dict[fields[0]] = fields[1].strip()
                        else:
                            self.logger.warning("Config file specifies unknown STWR type. Line will be ignored: {}".format(line))
        return rw_path_dict
