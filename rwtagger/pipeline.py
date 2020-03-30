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

    def predict(self, inputdir, outputdir, rwtype_list, input_format, chunk_len, test_scores, confidence_vals):
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

        # for the first rwtype:
        # initialize the tagger for this STWR type
        self.logger.info("Start predicting {}  (inputdir: {}, outputdir: {})".format(rwtype_list[0], inputdir, outputdir))
        # if it already exists, remove the outputdir and create an new one
        if os.path.exists(outputdir):
            try:
                shutil.rmtree(outputdir)
            except PermissionError:
                self.logger.error("Could not remove old output directory: {}.\nPlease remove manually. Tagging aborted.".format(outputdir))
                exit(0)
        os.makedirs(outputdir)
        rwtagger = tagger.RWTagger(self.use_gpu, self.log_level)
        rwtagger.predict(inputdir, outputdir, rwtype_list[0], input_format=input_format, chunk_len=chunk_len,
                         test_scores=test_scores,
                         output_confidence=confidence_vals)
        self.logger.info("Finished predicting {}".format(rwtype_list[0]))
        # for all following rwtypes: input is always tsv and tempdir is used as inputdir
        curr_path = os.path.dirname(os.path.abspath(__file__))
        tempdir = os.path.join(curr_path, "temp")
        self.logger.info("Creating tempdir: {}".format(tempdir))
        if len(rwtype_list) > 1:
            # remove any old tempdirs and create an new one
            if os.path.exists(tempdir):
                try:
                    shutil.rmtree(tempdir)
                except PermissionError:
                    self.logger.error("Could not remove old temp directory: {}.\nPlease remove manually. Tagging aborted.".format(tempdir))
                    exit(0)
            os.makedirs(tempdir)
            for rwtype in rwtype_list[1:]:
                # move all the tsv files from outputdir to temp dir
                tsv_files = [x for x in os.listdir(outputdir) if x[-4:] == ".tsv"]
                for file in tsv_files:
                    shutil.move(os.path.join(outputdir, file), os.path.join(tempdir, file))
                self.logger.info("Start predicting {} (inputdir: {}, outputdir: {})".format(rwtype, tempdir, outputdir))
                # initialize the tagger for this STWR type
                rwtagger = tagger.RWTagger(self.use_gpu, self.log_level)
                # inputtype in always "tsv" now and input_dir is the temp dir
                rwtagger.predict(tempdir, outputdir, rwtype, input_format="tsv", chunk_len=chunk_len, test_scores=test_scores,
                                 output_confidence=confidence_vals)
                self.logger.info("Finished predicting {}".format(rwtype))
            # when all types have been predicted, remove the temp dir
            try:
                shutil.rmtree(tempdir)
            except PermissionError:
                self.logger.warning("Predictions finished, but could not remove temp directory: {}.".format(tempdir))