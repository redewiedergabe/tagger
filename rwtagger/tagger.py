import flair
from flair.data import Sentence
from flair.data import Corpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter
from flair.embeddings import BertEmbeddings, WordEmbeddings, CharacterEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.data import Token

import sklearn.metrics
import os
import pandas as pd
import re
import datetime
import torch
import logging

import nltk.data
from nltk.tokenize import word_tokenize
import numpy as np


class RWTagger:
    def __init__(self, use_gpu, log_level):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        if use_gpu:
            flair.device = torch.device('cuda:0')  # or 'cuda:1' etc. depending on your setup...
        else:
            flair.data = torch.device('cpu')


    def _get_embeddings(self, embtype="bert"):
        """
        *** This method is only needed when training your own models. It is not accessible from
        rwtagger_script and not documented in detail. Use at your own risk. ;-)
        If you want to use this method, you need to have access to the appropriate language
        embeddings and must adjust the paths accordingly ***

        get the embedding combination, specified by the value of embtype
        Specify new embedding types here 
        :param embtype: 
        :return: Tuple for embedding name and embedding (flair format)
        """
        if embtype == "bert":
            # BERT embeddings
            _emb_name = "BERT"
            _embeddings = BertEmbeddings('bert-base-multilingual-cased')

        elif embtype == "fasttext":
            # fasttext out of the box
            _emb_name = "Fasttext"
            _embeddings = WordEmbeddings('de')

        elif embtype == "flair_stacked_skipgram300":
            # Stacked Embeddings (Fasttext rwk_cbow_100 und Flair)
            _emb_name = "Stacked: Fasttext (rwk_skipgram_300), Flair ('german-forward'), Flair ('german-backward')"
            _embeddings = StackedEmbeddings([
                WordEmbeddings("rwk_embeddings/rwk_embeddings_skipgram_300.gensim"),
                FlairEmbeddings('german-forward'),
                FlairEmbeddings('german-backward'),
            ])

        # with custom FLAIR_embeddings
        elif embtype == "flair_stacked_rw-flair-emb_skipgram300":
            # Stacked Embeddings (Fasttext rwk_cbow_100 und Flair)
            _emb_name = "Stacked: Fasttext (rwk_skipgram_300), rwk Flair forward, rwk Flair backward"
            _embeddings = StackedEmbeddings([
                WordEmbeddings("rwk_embeddings/rwk_embeddings_skipgram_300.gensim"),
                FlairEmbeddings('rwk_flair_embeddings/flair_embedding_model_forward.pt'),
                FlairEmbeddings('rwk_flair_embeddings/flair_embedding_model_backward.pt'),
            ])

        elif embtype == "flair_stacked_skipgram300_no-grenz":
            _emb_name = "Stacked: Fasttext (rwk_skipgram_300_ohne_grenz), Flair ('german-forward'), Flair ('german-backward')"
            _embeddings = StackedEmbeddings([
                WordEmbeddings("rwk_embeddings/rwk_embeddings_skipgram_300_ohne_grenz.gensim"),
                FlairEmbeddings('german-forward'),
                FlairEmbeddings('german-backward'),
            ])

        elif embtype == "flair_stacked_de_and_skipgram300_no-grenz":
            _emb_name = "Stacked: Fasttext (de) Fasttext (rwk_skipgram_300_ohne_grenz), Flair ('german-forward'), Flair ('german-backward')"
            _embeddings = StackedEmbeddings([
                WordEmbeddings("de"),
                WordEmbeddings("rwk_embeddings/rwk_embeddings_skipgram_300_ohne_grenz.gensim"),
                FlairEmbeddings('german-forward'),
                FlairEmbeddings('german-backward'),
            ])
           
        # with custom FLAIR embeddings
        elif embtype == "flair_stacked_rw-flair-emb_skipgram300_no-grenz":
            _emb_name = "Stacked: Fasttext (rwk_skipgram_300_no-grenz), rwk Flair forward, rwk Flair backward"
            _embeddings = StackedEmbeddings([
                WordEmbeddings("rwk_embeddings/rwk_embeddings_skipgram_300_ohne_grenz.gensim"),
                FlairEmbeddings('rwk_flair_embeddings/flair_embedding_model_forward.pt'),
                FlairEmbeddings('rwk_flair_embeddings/flair_embedding_model_backward.pt'),
            ])

        elif embtype == "flair_stacked_skipgram500_no-grenz":
            _emb_name = "Stacked: Fasttext (de) Fasttext (rwk_skipgram_300_ohne_grenz), Flair ('german-forward'), Flair ('german-backward')"
            _embeddings = StackedEmbeddings([
                WordEmbeddings("rwk_embeddings/rwk_embeddings_skipgram_500_ohne_grenz.gensim"),
                FlairEmbeddings('german-forward'),
                FlairEmbeddings('german-backward'),
            ])

            # with custom FLAIR embeddings
        elif embtype == "flair_stacked_rw-flair-emb_skipgram500_no-grenz":
            # Stacked Embeddings (Fasttext rwk_cbow_100 und Flair)
            _emb_name = "Stacked: Fasttext (rwk_skipgram_500_no-grenz), rwk Flair forward, rwk Flair backward"
            _embeddings = StackedEmbeddings([
                WordEmbeddings("rwk_embeddings/rwk_embeddings_skipgram_500_ohne_grenz.gensim"),
                FlairEmbeddings('rwk_flair_embeddings/flair_embedding_model_forward.pt'),
                FlairEmbeddings('rwk_flair_embeddings/flair_embedding_model_backward.pt'),
            ])

        else:
            print("Unknown embedding type: '{}'. Abort.".format(embtype))
            exit(1)
        return (_emb_name, _embeddings)

    def convert_txtfile_to_dateframe(self, txt_file):
        """
        uses NLTK to perform tokenization and sentence splitting on a txt file
        outputs a pandas dataframe with columns 'tok' and 'sentstart'
        :param txt_file:
        :return:
        """
        result = {"tok": [], "sentstart": []}
        # nltk sentence splitter
        sent_detector = nltk.data.load("tokenizers/punkt/german.pickle")

        with open(txt_file, "r", encoding="utf8") as f:
            content = "".join(f.readlines())
        sentences = sent_detector.tokenize(content.strip())
        for sent in sentences:
            # tokenize the sentence
            sent_tokens = word_tokenize(sent)
            result["tok"].append(sent_tokens[0])
            result["sentstart"].append("yes")
            for tok in sent_tokens[1:]:
                result["tok"].append(tok)
                result["sentstart"].append("no")
        result_df = pd.DataFrame(result)
        return result_df

    def create_sentlist_from_file_batchmax(self, data, maxlen=64, compare_column="cat"):
        """
        takes a pandas dataframe with columns 'tok' and 'sentstart' and creates a list of flair Sentence objects with tags.
        Each flair Sentence object may contain several real sentences, but at most maxlen tokens.
        The Sentence object stops at a sentence boundary, so it is often shorter than maxlen.
        Sentences longer than maxlen are split!
        If a line with token value "EOF" is encountered, a shorter flair Sentence object is returned,
        so no file boundaries are crossed
        :param data_path:
        :return:
        """
        sent_list = []
        toklist = []
        catlist = []
        # the len_last_token is needed to add proper start/end pos for each sentence token
        len_last_token = 0
        # track the sentence that is currently being processed
        curr_sentence_tok = []
        curr_sentence_cat = []
        for index, row in data.iterrows():
            tok = str(row["tok"])
            if compare_column != "NaN":
                cat = str(row[compare_column])
            else:
                cat = "-"

            # if the current token is "EOF" this marks the end of sample file
            # chunks may not cross file boundaries, therefore end the sentence here in any case
            if tok == "EOF":
                # do not add this token to any list
                # merge toklist and curr_sentence_tok list to get all current tokens
                # and create a flair sentence
                toklist.extend(curr_sentence_tok)
                catlist.extend(curr_sentence_cat)
                self.logger.debug("create chunk at EOF with (len: {}): {}".format(len(toklist), toklist))
                self.logger.debug("catlist with (len: {}): {}".format(len(catlist), catlist))
                sent = Sentence()
                for i, tok in enumerate(toklist):
                    flair_tok = Token(str(tok), start_position=len_last_token)
                    len_last_token += len(tok) + 1
                    flair_tok.add_tag("cat", catlist[i])
                    sent.add_token(flair_tok)
                if len(sent.tokens) > 0:
                    sent_list.append(sent)
                len_last_token = 0
                toklist = []
                catlist = []
                # reset the curr sent lists as well
                curr_sentence_tok = []
                curr_sentence_cat = []

            else:
                # if we are at the start of a new sentence, add the contents of curr_sentence_tok
                # and curr_sentence_cat to the main lists and start a new curr_sentence
                if row["sentstart"] == "yes":
                    toklist.extend(curr_sentence_tok)
                    catlist.extend(curr_sentence_cat)
                    curr_sentence_tok = [tok]
                    curr_sentence_cat = [cat]
                else:
                    curr_sentence_tok.append(tok)
                    curr_sentence_cat.append(cat)

                # if the combined length of toklist and curr_sentence_tok is > maxlen now,
                # create a flair sentence with the tokens in toklist and reset it
                # the remaining tokens in curr_sentence_tok are saved for the next chunk
                if len(toklist) + len(curr_sentence_tok) > maxlen:
                    # if toklist is empty at this point, we have a sentence > maxlen
                    # and must split it. The last token currently in curr_sentence will
                    # be preserved for later so that the chunk is not too long
                    if len(toklist) == 0:
                        toklist.extend(curr_sentence_tok[0:-1])
                        catlist.extend(curr_sentence_cat[0:-1])
                        curr_sentence_tok = [curr_sentence_tok[-1]]
                        curr_sentence_cat = [curr_sentence_cat[-1]]
                        self.logger.debug("Sentence is split (len: {}): {}".format(len(toklist), toklist))

                    self.logger.debug("create chunk with (len: {}): {}".format(len(toklist), toklist))
                    self.logger.debug("catlist with (len: {}): {}".format(len(catlist), catlist))
                    sent = Sentence()
                    for i, tok in enumerate(toklist):
                        flair_tok = Token(str(tok), start_position=len_last_token)
                        len_last_token += len(tok) + 1
                        flair_tok.add_tag("cat", str(catlist[i]))
                        sent.add_token(flair_tok)
                    if len(sent.tokens) > 0:
                        sent_list.append(sent)
                    len_last_token = 0
                    toklist = []
                    catlist = []

        self.logger.debug("toklist: {}, curr_sent_tok: {}".format(len(toklist), len(curr_sentence_tok)))
        # if the loop is complete, empty the buffers and add them to the list
        if len(curr_sentence_tok) > 0:
            toklist.extend(curr_sentence_tok)
            catlist.extend(curr_sentence_cat)
            sent = Sentence()
            for i, tok in enumerate(toklist):
                flair_tok = Token(str(tok), start_position=len_last_token)
                len_last_token += len(tok) + 1
                flair_tok.add_tag("cat", str(catlist[i]))
                sent.add_token(flair_tok)
            if len(sent.tokens) > 0:
                sent_list.append(sent)
            len_last_token = 0

        return sent_list


    def create_corpus(self, train_path, val_path, test_path, chunk_len):
        """
        *** This methods is only needed when training your own models
        It is not accessible from rwtagger_script and not documented in detail. Use at your own risk. ;-)
        ***
        :param data_path:
        :return:
        """
        train_list = self.create_sentlist_from_file_batchmax(train_path, maxlen=chunk_len)
        val_list = self.create_sentlist_from_file_batchmax(val_path, maxlen=chunk_len)
        test_list = self.create_sentlist_from_file_batchmax(test_path, maxlen=chunk_len)
        corpus: Corpus = Corpus(train_list, val_list, test_list)

        return corpus

    def train(self, trainfile, devfile, testfile, resfolder, embtype="bert", chunk_len=100, batch_len=8):
        """
        *** This method can be used to train new models with the settings used in project Redewiedergabe
        It is not accessible from rwtagger_script and not documented in detail. Use at your own risk. ;-)
        ***
        :param trainfile:
        :param devfile:
        :param testfile:
        :param resfolder:
        :param embtype:
        :param chunk_len:
        :param batch_len:
        :return:
        """
        emb_name, embeddings = self._get_embeddings(embtype)
        
        corpus: Corpus = self.create_corpus(trainfile, devfile, testfile, chunk_len)
        tag_dictionary = corpus.make_tag_dictionary(tag_type="cat")

        if not os.path.exists(resfolder):
            os.makedirs(resfolder)

        tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                                embeddings=embeddings,
                                                tag_dictionary=tag_dictionary,
                                                tag_type="cat",
                                                use_crf=True,
                                                rnn_layers=2
                                                )
        trainer: ModelTrainer = ModelTrainer(tagger, corpus)

        trainer.train(resfolder,
                      learning_rate=0.1,
                      mini_batch_size=batch_len,
                      max_epochs=150,
                      checkpoint=True)
        # plot training curves
        plotter = Plotter()
        plotter.plot_training_curves(os.path.join(resfolder, 'loss.tsv'))
        plotter.plot_weights(os.path.join(resfolder, 'weights.txt'))

    def predict(self, input_dir, output_dir, rw_type, input_format, chunk_len=100,
                test_scores = False,
                output_confidence=False):
        """
        tags each file in the input directory (txt or tsv files) and writes the results
        to output_dir. Also adds a folder "result_stats" with runtime information to the
        output_dir

        tsv files must have at least the columns 'tok' and 'sentstart'
        :param input_dir: string value: path to input directory
        :param output_dir: string value: path to output directory
        :param rw_type: string value: direct, indirect, freeIndirect or reported
        :param input_format: string value: txt or tsv
        :param chunk_len:
        :return:
        """
        # time the prediction
        start_time = datetime.datetime.now().replace(microsecond=0)
        # create a subdir for testing and overview information in the outputdir
        result_subdir = "result_stats"
        if not os.path.exists(os.path.join(output_dir, result_subdir)):
            os.makedirs(os.path.join(output_dir, result_subdir))

        # load the model
        # determine the current script path
        curr_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(curr_path, "models", rw_type, "final-model.pt")
        if not os.path.exists(model_path):
            logging.warning("Predicting {} aborted. Model not found at path '{}'. Please download a model and put it into "
                          "the appropriate directory. The model file must be named final-model.pt.".format(rw_type, model_path))
        else:
            self.logger.info("loading model {}".format(model_path))
            model = SequenceTagger.load(model_path)
            self.logger.info("model loaded")

            # if test mode, collect score data (initialize in any case)
            score_dict = {"file": [], "f1":[], "precision": [], "recall": []}
            all_predictions_df = pd.DataFrame()

            input_files = [x for x in os.listdir(input_dir)]
            for file in input_files:
                resfile_name = re.sub("\..+$", ".tsv", file)
                self.logger.info("predicting {}".format(file))
                # read the file and convert to dataframe
                if input_format == "txt":
                    data = self.convert_txtfile_to_dateframe(os.path.join(input_dir, file))
                else:
                    data = pd.read_csv(os.path.join(input_dir, file), sep="\t", quoting=3, encoding="utf-8", na_values=[])

                # check for tok column:
                if "tok" not in data.columns:
                    self.logger.warning("Column 'tok' is missing in file {}. File will be skipped.".format(file))
                else:
                    if "sentstart" not in data.columns:
                        self.logger.warning("Column 'sentstart' is missing in file {}. Will be added with default values (all 'no').".format(file))
                        data["sentstart"] = ["no"]*len(data)

                    self.logger.debug("TEST: data head:\n {}".format(data.head(10)))
                    # create sentlist (based on max chunk length)
                    sent_list = self.create_sentlist_from_file_batchmax(data,
                                                                        maxlen=chunk_len,
                                                                        compare_column="NaN")
                    # predict
                    res_dict = {"tok": [], rw_type + "_pred": [], rw_type + "_conf": []}
                    for sent in sent_list:
                        model.predict(sent)
                        pred_list = [x["type"] for x in sent.to_dict("cat")["entities"]]
                        res_dict["tok"].extend([x["text"] for x in sent.to_dict("cat")["entities"]])
                        res_dict[rw_type + "_conf"].extend([x["confidence"] for x in sent.to_dict("cat")["entities"]])
                        res_dict[rw_type + "_pred"].extend(pred_list)
                    pred_df = pd.DataFrame(res_dict)
                    # create output
                    # if there is a missmatch in file length after prediction, still save the results
                    if (len(data) != len(pred_df)):
                        self.logger.warning("File length changed when predicting for file {} (before: {}, after: {})\n"
                                        "Result file will be saved with prefix 'warn_'; additional columns are lost."
                                      .format(file, len(data), len(pred_df)))
                        pred_df.to_csv(os.path.join(output_dir, "warn_" + resfile_name), index=False, sep="\t")
                    # if everything is okay, add the new column(s) to the original data and save
                    else:
                        if output_confidence:
                            data[rw_type + "_conf"] = pred_df[rw_type + "_conf"]
                        data[rw_type + "_pred"] = pred_df[rw_type + "_pred"]
                        data.to_csv(os.path.join(output_dir, resfile_name), index=False, sep="\t", encoding="utf-8")
                        # calculate the testscores:
                        if test_scores:
                            self.logger.info("Calculate scores for {}".format(file))
                            if rw_type in data.columns and rw_type + "_pred" in data.columns:
                                data, f1, prec, rec = self.calculate_scores(data, rw_type)
                                score_dict["file"].append(file)
                                score_dict["f1"].append(f1)
                                score_dict["precision"].append(prec)
                                score_dict["recall"].append(rec)
                                all_predictions_df = all_predictions_df.append(data)
                            else:
                                self.logger.warning("Skipping test scores for file {}: Missing column {} and/or {}".format(file, rw_type, rw_type + "_pred"))

            end_time = datetime.datetime.now().replace(microsecond=0)

            # write an overview file when the process is finished
            res_text = "RW Tagger (predict): Model {}\n" \
                       "Predict time:\nstart: {}nend:{}\ntotal: {}" \
                .format(model_path, start_time, end_time, end_time - start_time)
            # if in test mode, calculate the final scores (for all the data) and save the test score df
            if test_scores:
                self.logger.info("Calculate total scores")
                if len(all_predictions_df) > 0:
                    self.logger.debug("all_predictions_len: {}".format(len(all_predictions_df)))
                    all_predictions_df, f1, prec, rec = self.calculate_scores(all_predictions_df, rw_type)
                    score_dict["file"].append("total")
                    score_dict["f1"].append(f1)
                    score_dict["precision"].append(prec)
                    score_dict["recall"].append(rec)
                    score_df = pd.DataFrame(score_dict)
                    score_df.to_csv(os.path.join(output_dir, result_subdir, rw_type + "_test_scores.tsv"), index=False, sep="\t", encoding="utf-8")
                    res_text += "\nTotal test scores (for detailed scores see {}_test_scores.tsv):\n" \
                                "f1: {}, precision: {}, recall: {}".format(rw_type, f1, prec, rec)
                    self.logger.info("Total scores for {}: f1: {}, precision: {}, recall: {}".format(rw_type, f1, prec, rec))
            with open(os.path.join(output_dir, result_subdir, rw_type + "_overview.txt"), "w", encoding="utf-8") as f:
                f.write(res_text)


    def calculate_scores(self, data, rw_type):
        """
        expects data with column names <rwtype> and <rwtype>_pred
        determines the status of each prediction; calculates f1, precision, recall
        returns data with new column <rwtype>_status and f1, prec, recall values
        :param data: dataframe with column names <rwtype> and <rwtype>_pred
        :param rw_type:
        :return: (data, f1, precision, recall)
        """
        # check whether the data has the right format; if not return data as if and 0 for
        # all scores
        if rw_type in data.columns and rw_type + "_pred" in data.columns:
            real_list = list(data[rw_type])
            pred_list = list(data[rw_type + "_pred"])
            status_list = []

            for i, real_val in enumerate(real_list):
                pred_val = pred_list[i]
                status = "correct-x"
                if real_val == pred_val:
                    if real_val != "x":
                        status = "correct"
                else:
                    if real_val == "x":
                        status = "FP"
                    else:
                        status = "FN"
                status_list.append(status)
            data[rw_type + "_status"] = status_list

            if (rw_type not in pred_list) and (rw_type not in real_list):
                self.logger.info("STWR type '{}' not found in both real and predicted values. Scores are set to 1.".format(rw_type))
                fscore = 1.0
                precision = 1.0
                recall = 1.0
            else:
                fscore = np.round(sklearn.metrics.f1_score(real_list, pred_list, pos_label=rw_type), 2)
                precision = np.round(sklearn.metrics.precision_score(real_list, pred_list, pos_label=rw_type), 2)
                recall = np.round(sklearn.metrics.recall_score(real_list, pred_list, pos_label=rw_type), 2)
        else:
            fscore = 0.0
            precision = 0.0
            recall = 0.0

        return(data, fscore, precision, recall)
