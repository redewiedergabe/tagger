# Recognizers for German speech, thought and writing representation (STWR)

<p>
<a href="https://console.tiyaro.ai/explore/redewiedergabe-bert-base-historical-german-rw-cased"> <img src="https://tiyaro-public-docs.s3.us-west-2.amazonaws.com/assets/try_on_tiyaro_badge.svg"></a>
</p>

---
**NOTE**: 
The code was adjusted to account for new versions of the **pytorch** and **flair** packages. Please note the new [requirements](requirements.txt) - especially: 
* pytorch version 1.10.1 
* flair version 0.10

The code will no longer work with the old requirements. The old version of the tagger code has been archived as **release v1.0.0**. 

Please use Github's issue tracker if you encounter any problems.

---

These recognizers were developed by the DFG-funded project "Redewiedergabe - eine literatur- und sprachwissenschaftliche Korpusanalyse" (Leibniz Institute for the German Language / University of Würzburg) ([www.redewiedergabe.de](http://redewiedergabe.de)) and (mostly) trained on data from [Corpus Redewiedergabe](https://github.com/redewiedergabe/corpus). 

They can be used to automatically detect and annotate the following 4 types of speech, thought and writing representation in German texts.

| STWR type                      | Example                                                                 | Translation                                              |
|--------------------------------|-------------------------------------------------------------------------|----------------------------------------------------------|
| direct                         | Dann sagte er: **"Ich habe Hunger."**                                       | Then he said: **"I'm hungry."**                             |
| free indirect ('erlebte Rede') | Er war ratlos. **Woher sollte er denn hier bloß ein Mittagessen bekommen?** | He was at a loss. **Where should he ever find lunch here?** |
| indirect                  | Sie fragte, **wo das Essen sei.**                                           | She asked **where the food was.**                            |
| reported                  | **Sie sprachen über das Mittagessen.**                                      | **They talked about lunch.**                                 |

For more detailed descriptions of these STWR types please refer to the [Redewiedergabe annotation guidelines](http://redewiedergabe.de/richtlinien/richtlinien.html) (in German).

The recognizers are based on deepLearning and utilize the [FLAIR NLP framework](https://github.com/flairNLP). 

### Publications
Main Publication (please cite when using the recognizers):

**Annelen Brunner, Ngoc Duyen Tanja Tu, Lukas Weimer, Fotis Jannidis: [To BERT or not to BERT – Comparing contextual embeddings in a deep learning architecture for the automatic recognition of four types of speech, thought and writing representation](http://ceur-ws.org/Vol-2624/paper5.pdf), Proceedings of the 5th Swiss Text Analytics Conference (SwissText) & 16th Conference on Natural Language Processing (KONVENS), Zurich, Switzerland, June 23-25, 2020.**

Other Publications:

Annelen Brunner, Ngoc Duyen Tanja Tu, Lukas Weimer, Fotis Jannidis: [Deep learning for Free Indirect Representation](https://corpora.linguistik.uni-erlangen.de/data/konvens/proceedings/papers/KONVENS2019_paper_27.pdf), KONVENS Erlangen 2019, pp. 241-245.

### Quick links
* [Get our models](#recognizer-models)
* [Get our custom-trained language embeddings (BERT and FLAIR)](#custom-trained-language-embeddings)
* [Set up the recognizer](#recognizer-setup) 
    * **Note**: As an alternative to using the code provided here, you can also use run our models [directly via the FLAIR framework](https://github.com/flairNLP/flair/issues/1531)! 

## Recognizer models 

[Current top models](#top-models-konvens-2020)

Each STWR type is recognized by a separate model.
The downloads are zip archives. Simply unpack them and move the folders into directory **rwtagger/models**.

All models are named **final-model.pt** and the name of the subfolder is used to locate the model you want to use. 
The default models are stored in directories named after their STWR type
(**direct**, **indirect**, **reported** or **freeIndirect**).

The subfolders for alternative models have different names. If you want to
use them, you have to edit the file **rwtagger/config.txt**

**Example:** For using the [direct model that is based on BERT embeddings](#alternative-models-konvens-2020), first download and unpack this alternative model. It is stored in a folder namend **direct_BERT**. Move the folder into the directory **rwtagger/models**. Then add the following line to the file **rwtagger/config.txt**:

 ```direct@direct_BERT```  
 
When you run the rwtagger script again, the BERT model will be used to recognize direct STWR instead of the default.

### KONVENS 2020 models
These are the models discussed in the [KONVENS 2020 paper](http://ceur-ws.org/Vol-2624/paper5.pdf). 

All models first encode the text with a [customized Language Embedding](#custom-trained-language-embeddings) (depending on the model: see table) and were then trained for their STWR task using a deep learning architecture with 2 BiLSTM layers and one CRF layer. 

The recognizers work on token basis and the scores are calculated based on tokens as well.

Each model recognizes one specific type of STWR in a binary classification ("direct" vs. "x", "indirect" vs. "x", etc.).

The training, validation and test corpora used to train and evaluate the taggers 
for direct, indirect and reported STWR are available [here](https://github.com/redewiedergabe/corpus/blob/master/resources/docs/data_konvens-paper-2020.md). Unfortunately, we cannot provide the exact data for the free indirect model due to copyright restrictions.

#### Top models (KONVENS 2020)

These are the best performing models as presented in the [KONVENS 2020 paper](http://ceur-ws.org/Vol-2624/paper5.pdf). They are considered the default models for the recognizers.

[Package with all 4 STWR models at once](http://www.redewiedergabe.de/models/models.zip) (~3 GB)

| STWR type     | F1   | Precision | Recall | Language embedding                                   | Training and Test material                                                     | Download |
|---------------|------|-----------|--------|------------------------------------------------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------|
| direct        | 0.85 | 0.93      | 0.78   | Skipgram with 500 dimensions  & FLAIR embeddings (both custom trained) | historical German (19th to early 20th century), fiction and non-fiction | [Direct model](http://www.redewiedergabe.de/models/direct.zip) (~1.6 GB) |
| indirect      | 0.76 | 0.81      | 0.71   | BERT (custom finetuned)                              | historical German (19th to early 20th century), fiction and non-fiction | [Indirect model](http://www.redewiedergabe.de/models/indirect.zip) (~460 MB)
| reported      | 0.60 | 0.67      | 0.54   | BERT (custom finetuned)                              | historical German (19th to early 20th century), fiction and non-fiction | [Reported model](http://www.redewiedergabe.de/models/reported.zip) (~460 MB) |
| free indirect | 0.59 | 0.78      | 0.47   | BERT (custom finetuned)                              | historical and modern German (late 19th century to current), only fiction             |[Free indirect model](http://www.redewiedergabe.de/models/freeIndirect.zip) (~460 MB)|

#### Alternative models (KONVENS 2020)
As an alternative, we also provide the most successful models using an alternative language embedding. These were used in the comparisons in the [KONVENS 2020 paper](http://ceur-ws.org/Vol-2624/paper5.pdf).

| STWR type     | F1   | Precision | Recall | Language embedding                                   | Training and Test material                                                     | Download |
|---------------|------|-----------|--------|------------------------------------------------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------|
| direct        | 0.80 | 0.87      | 0.74   | BERT (custom finetuned)  | historical German (19th to early 20th century), fiction and non-fiction | [Direct model](http://www.redewiedergabe.de/models/direct_BERT.zip) (~ 460 MB) |
| indirect      | 0.74 | 0.77      | 0.71   |  Skipgram with 300 dimensions  & FLAIR embeddings (both custom trained)                              | historical German (19th to early 20th century), fiction and non-fiction | [Indirect model](http://www.redewiedergabe.de/models/indirect_FLAIR.zip) (~788 MB)
| reported      | 0.58 | 0.69      | 0.50   |  Skipgram with 500 dimensions  & FLAIR embeddings (both custom trained)                              | historical German (19th to early 20th century), fiction and non-fiction | [Reported model](http://www.redewiedergabe.de/models/reported_FLAIR.zip) (~1.6 GB) |
| free indirect | 0.51 | 0.87      | 0.36   |  Skipgram with 300 dimensions  & FLAIR embeddings (both custom trained)                          | historical and modern German (late 19th century to current), only fiction             |[Free indirect model](http://www.redewiedergabe.de/models/freeIndirect_FLAIR.zip) (~788 MB)|
 

## Custom-trained language embeddings
Historical German (19th to early 20th century) (fiction and non-fiction) was used for customizing/finetuning the Language Embeddings used for the recognizer modules.
* The fine-tuned BERT Model is available at huggingface.co: https://huggingface.co/redewiedergabe/bert-base-historical-german-rw-cased
* The custom-trained FLAIR model was integrated into the [FLAIR framework](https://github.com/flairNLP/flair/issues/1502) and can be loaded with the following lines of code:
```embeddings = FlairEmbeddings('de-historic-rw-forward')``` and ```embeddings = FlairEmbeddings('de-historic-rw-backward')```

# Recognizer setup

This GitHub repository contains scripts that handle data input and output and optionally calculate test scores. They allow you to run the recognizers from the command line.

For this, a Python environment with the necessary modules has to be set up. We provide a requirements file and give some instructions how to set up a [Python virtual environment](#environment) to facilitate this. 

The [trained models](#recognizer-models) must be downloaded separately before the recognizers are usable. Put them into the directory **rwtagger/models**. Models must always be named **final-model.pt** and be stored in a sub-folder matching their type (**direct**, **indirect**, **reported** or **freeIndirect**).

## Environment
The software was developed with Python 3.7.0., but also runs with Python 3.6.8 and should work for newer Python versions as well.

The following instructions explain how to set up the necessary Python modules in a virtual enviroment. Of course you can also execute the recognizers in your regular Python environment, if the necessary modules are installed there. 

We cannot cover all variations of the setup of a Python virtual  environment here, so the instructions explain only how to do it with Anaconda Python under Windows. If this does not work for you, please consult other instructions regarding Python virtual environments, e.g. [this one](https://docs.python.org/3/library/venv.html).

**Setup with Anaconda Python under Windows**
* Make sure the you have at least Python 3.6.8 installed (3.7.0 is recommended; newer versions should work as well) 
* If you have no experience with Python, we recommend installing [Anaconda Python](https://www.anaconda.com/); then proceed in the 'Anaconda Powershell Prompt' console to avoid problems with path variables (NOTE: Anaconda has two different 'Prompt' consoles. These instructions assume you use 'Anaconda **Powershell** Prompt')
* If Python **virtualenv** is not already installed, execute the following code in the console:
  
  ```pip install virtualenv```
* Download and unpack this Github project
* Change into the directory **tagger** and execute the following code 
   * **NOTE:** The code below installs the **CPU version of pytorch**, which works for all computers. If you want to use a GPU instead, refer to the [pytorch download page](https://pytorch.org/get-started/locally) to get the correct syntax. However, for the GPU to work with pytorch your also have to make sure you have CUDA installed and configured correctly. For this, please refer to other guides, e.g. [this one](https://medium.com/datadriveninvestor/installing-pytorch-and-tensorflow-with-cuda-enabled-gpu-f747e6924779).
   ```
   virtualenv venv 
   cd venv
   .\Scripts\activate
   # --> You should now see '(base) (venv)' at the beginning of your prompt line.
   
   # Install pytorch. The code was tested with version 1.10.1. 
   # If you don't want to use a GPU and are willing to risk getting a newer version, simply type:
   pip3 install torch torchvision torchaudio
   # Alternatively, if you want more control over the version or want to use a GPU, skip 
   # the line above and refer to https://pytorch.org/get-started/locally to get the correct 
   # syntax for installation. 
   
   # Install all other required modules:
   pip install -r ..\requirements.txt
   
   # Change back to the rwtagger directory
   cd ..\rwtagger
  ```
* To tokenize input texts, you need additional libraries for the NLTK module. We recommend installing them in the interactive mode:
  * type ```python``` to open the Python interpreter. Then type the following:
```
  import nltk
  nltk.download('punkt')
  exit()
```
* You can now execute the recognizers in this console window (after you have downloaded the Recognizer models). Make sure that **venv** is the active environment (should be visible in your prompt line). If you want to switch back to your regular Python environment, change back to venv/Scripts and type:

   ```deactivate```

## First steps
After setting up the module and putting the models into the appropriate model folders, you can use the recognizers to annotate your texts. For this, execute the script **rwtagger.py** in your console.

This script can be used to annotate textual data with the STWR types *direct*, *freeIndirect*, 
*indirect* and *reported*. It runs on CPU by default, but can use GPU if flag -gpu is specified (and your pytorch installation is properly set up to use the GPU). It the flag -conf is set, confidence values for the annotations are given as well. 

Data input can be plain text or tsv files (encoding: UTF-8 in both cases). Tsv files must contain a tab-separated column format with one token per line and two columns: Column 'tok' contains the tokens and is mandatory. Column 'sentstart' codes sentence boundaries ('yes' for the first token of a sentence, 'no' otherwise). If this column is missing, the text is treated like one sentence.

Result data will always be in tsv format. You can use the script **util/tsv_to_excel.py** to convert tsv files into excel files for convenience.

To view the program help, execute ```python rwtagger.py -h``` 

### Predict mode (-m predict)
In this mode, the script simply predicts the category for each token in the input files. The results are written into a column named after the predicted catgory (e.g. 'direct_pred').

Some statistics such as running time are written to a folder called 'result_stats' in your output directory.
  
### Test mode (-m test)

In this mode, the script does the same as in predict mode, but additionally calculates f1 scores, recall and precision between the predicted values and a gold standard. 

Input format must a tsv file with a column for each STWR type you want to calculate scores for. For example, for direct, the file must contain a column named 'direct'. For each token, the value must be either 'direct' (positive case) or 'x' (negative case) (just like the output format of the recognizer). The script adds an additional column 'direct_pred' and calculates the scores between those columns. A detailed analysis is written to a folder called 'result_stats' in your output directory.

You can use the script **util/create_testformat_from_rwcorpus.py** to convert any files that are in the column-based-text format of the [corpus Redewiedergabe](https://github.com/redewiedergabe/corpus) into a format that allows you to use them as input file for the rwtagger in test mode. 


### Some examples
NOTE: It is safest to always place the option parameters *after* input_dir and output_dir.

The directory **test** contains some folders you can use for testing. Note that the data in the output folder will be overwritten whenever you call the script again.

```python rwtagger.py input_dir output_dir```  

simplest call: expects an input folder of plain text UTF-8 coded files, tags all 4 STWR types and outputs tsv files with columns for each type; runs the tagger on CPU (Note: This call might take some time, as it loads and executes all 4 taggers one after the other)
    
```python rwtagger.py input_dir output_dir -t direct indirect -conf```

annotates only the types *direct* and *indirect*; outputs confidence values for each annotation; expects an input folder of plain text UTF-8 coded files 
        
 ```python rwtagger.py input_dir output_dir -gpu -f tsv ``` 

runs the tagger on GPU; input format is not plain text but tsv (similar to the output format of the tagger: one token per line and markers for sentence start; column names must be 'tok' and 'sentstart'); annotates all 4 STWR types 


 ```python rwtagger.py input_dir output_dir -m test -t reported ``` 

runs the tagger and also calculates test scores for the STWR type reported; input files must be tsv format and contain a column called 'reported' containing the gold standard annotations.


