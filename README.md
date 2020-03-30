# Recognizers for German speech, thought and writing representation (STWR)

**NOTE**: This is the first release of the STWR recognizers. Please use Github's issue tracker if you encounter any problems.

These recognizers were developed by the DFG-funded project "Redewiedergabe - eine literatur- und sprachwissenschaftliche Korpusanalyse" (Leibniz Institute for the German Language / University of Würzburg) ([www.redewiedergabe.de](http://redewiedergabe.de)) and (mostly) trained on data from [Corpus Redewiedergabe](https://github.com/redewiedergabe/corpus). 

They can be used to automatically detect and annotate the following 4 types of speech, thought and writing representation in German texts.

| STWR type                      | Example                                                                 | Translation                                              |
|--------------------------------|-------------------------------------------------------------------------|----------------------------------------------------------|
| direct                         | Dann sagte er: **"Ich habe Hunger."**                                       | Then he said: **"I'm hungry."**                             |
| free indirect ('erlebte Rede') | Er war ratlos. **Woher sollte er denn hier bloß ein Mittagessen bekommen?** | He was at a loss. **Where should he ever find lunch here?** |
| indirect                  | Sie fragte, **wo das Essen sei.**                                           | She asked **where the food was.**                            |
| reported                  | **Sie sprachen über das Mittagessen.**                                      | **They talked about lunch.**                                 |

For more definitions for these STWR types please refer to the [Redewiedergabe annotation guidelines](http://redewiedergabe.de/richtlinien/richtlinien.html) (in German).

The recognizers are based on deepLearning and utilize the [flair NLP framework](https://github.com/flairNLP). 

This GitHub repository contains scripts that handle data input and output and optionally calculate test scores. They allow you to run the recognizers from the command line.

 For this, a Python environment with the necessary modules has to be set up. We provide a requirements file and give some instructions how to set up a Python virtual environment to facilitate this (see Environment). 

The trained models must be downloaded separately before the recognizers are usable (see Recognizer models). 

### Environment
The module is developed with Python 3.7.0., but should work for newer Python versions as well.

The following instructions explain how to set up the necessary Python modules in a virtual enviroment. Of course you can also execute the recognizers in your regular Python environment, if the necessary modules are installed there. 

We cannot cover all variations of the setup of a Python virtual  environment here, so the instructions explain only how to do it with Anaconda Python under Windows. If this does not work for you, please consult other instructions regarding Python virtual environments, e.g. [this tutorial](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/).

**Setup with Anaconda Python under Windows**
* Make sure the you have at least Python 3.7.0 installed (newer versions should work as well) 
* If you have no experience with Python, we recommend installing [Anaconda Python](https://www.anaconda.com/); then proceed in the 'Anaconda Powershell Prompt' console to avoid problems with path variables (NOTE: Anaconda has two different 'Prompt' consoles. These instructions assume you use 'Anaconda **Powershell** Prompt')
* If Python **virtualenv** is not already installed, execute the following code in the console:
  
  ```pip install virtualenv```
* Download this Github project
* Change into the directory **redewiedergabe** and execute the following code 
   * **NOTE:** The code below installs the **CPU version of pytorch**, which works for all computers. If you want to use a GPU instead, uncomment the alternative line in the code. However, for the GPU to work with pytorch your also have to make sure you have CUDA installed and configured correctly. For this, please refer to other guides, e.g. [this one](https://medium.com/datadriveninvestor/installing-pytorch-and-tensorflow-with-cuda-enabled-gpu-f747e6924779).
   ```
   virtualenv venv 
   cd venv
   .\Scripts\activate
   # --> you should now see '(base) (venv)' at the beginning of your prompt line
   # install pytorch:
   # if your computer does not have a GPU:
   pip3 install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
   # alternatively, if your computer has a GPU you want to use, remove the line above and uncomment the following:
   # pip3 install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html
   # install all other required modules:
   pip install -r ..\requirements.txt
   # change to the rwtagger directory
   cd ..\rwtagger
  ```
* To tokenize input texts, you need additional libraries for the NLTK module. We recommend installing them in the interactive mode:
  * type ```python``` to open the Python interpreter. Then type the following:
```
  import nltk
  nltk.download('punkt')
  exit()
```
* You can now execute the recognizers in this console window (after you have downloaded the Recognizer models). Make sure that **venv** is the active environment (should be visible in your prompt line). If you want to switch back to your regular Python environment, type:

   ```deactivate```

### Recognizer models 
For each STWR type, we provide the model that scored best in our tests (scores see below).

Download the models and put them into the directory **rwtagger/models**. Models must always be named **final-model.pt** and be stored in a sub-folder matching their type (**direct**, **indirect**, **reported** or **freeIndirect**).

The downloads are zip archives. Simply unpack them and move the folders into the **models** directory.

* [Package with all 4 STWR models](http://www.redewiedergabe.de/models/models.zip) (~3 GB)

Separate downloads for the different STWR types:
* [Direct model](http://www.redewiedergabe.de/models/direct.zip) (~1.6 GB)
* [Indirect model](http://www.redewiedergabe.de/models/indirect.zip) (~460 MB)
* [Reported model](http://www.redewiedergabe.de/models/reported.zip) (~460 MB)
* [Free indirect model](http://www.redewiedergabe.de/models/freeIndirect.zip) (~460 MB)

### Scores for the models
All models first encode the text with a language embedding (depending on the model: see table) and were then trained for their STWR task using a deep learning architecture with 2 BiLSTM layers and one CRF layer. 

Historical German (19th to early 20th century) (fiction and non-fiction) was used for customizing/finetuning the Language Embeddings. 

The recognizers work on a token basis and the scores are calculated based on tokens as well.

| STWR type     | F1   | Precision | Recall | Language embedding                                   | Training and Test material                                                     |
|---------------|------|-----------|--------|------------------------------------------------------|-------------------------------------------------------------------------|
| direct        | 0.85 | 0.93      | 0.78   | Skipgram with 500 dimensions  & FLAIR embeddings (both custom trained) | historical German (19th to early 20th century), fiction and non-fiction |
| indirect      | 0.76 | 0.81      | 0.71   | BERT (custom finetuned)                              | historical German (19th to early 20th century), fiction and non-fiction |
| reported      | 0.60 | 0.67      | 0.54   | BERT (custom finetuned)                              | historical German (19th to early 20th century), fiction and non-fiction |
| free indirect | 0.59 | 0.78      | 0.47   | BERT (custom finetuned)                              | modern German (mid 20th century to current), only fiction             |

# First steps
After setting up the module and putting the models into the appropriate model folders, you can use the recognizers to annotate your texts. For this, execute the script **rwtagger.py** in your console.

This script can be used to annotate textual data with the STWR types *direct*, *freeIndirect*, 
*indirect* and *reported*. It runs on CPU by default, but can use GPU if flag -gpu is specified (and your pytorch installation is properly set up to use the GPU). It the flag -conf is set, confidence values for the annotations are given as well. 

Data input can be plain text or tsv files (encoding: UTF-8 in both cases). Tsv files must contain a tab-separated column format with one token per line and two columns: Column 'tok' contains the tokens and is mandatory. Column 'sentstart' codes sentence boundaries ('yes' for the first token of a sentence, 'no' otherwise). If this column is missing, the text is treated like one sentence.

Result data will always be in tsv format. You can use the script **util/tsv_to_excel.py** to convert tsv files into excel files for convenience.

To view the program help, execute ```python rwtagger.py -h``` 

### Predict mode (-m predict)
In this mode, the script simply predicts the category for each token in the input files. The results are written into a column named after the predicted catgory (e.g. 'direct_pred').

 Information about some statistics such as running time is written to a folder called 'result_stats' in your output directory.
  
### Test mode (-m test)

In this mode, the script does the same as in predict mode, but additionally calculates f1 scores, recall and precision between the predicted values and a gold standard. 

Input format must a tsv file with a column for each STWR type you want to calculate scores for. For example, for direct, the file must contain a column named 'direct'. For each token, the value must be either 'direct' (positive case) or 'x' (negative case) (just like the output format of the recognizer). The script adds an additional column 'direct_pred' and calculates the scores between those columns. A detailed analysis is written to a folder called 'result_stats' in your output directory.

You can use the script **util/create_testformat_from_rwcorpus.py** to convert any files that are in the column-based-text format of the corpus REDEWIEDERGABE into a format that allows you to use them as input file for the rwtagger_script in test mode. 


### Some examples
The directory **test** contains some folders you can use for testing.

```python rwtagger.py input_dir output_dir```  

simplest call: expects an input folder of plain text UTF-8 coded files, tags all 4 STWR types and outputs tsv files with columns for each type; runs the tagger on CPU (Note: This call might take some time, as it loads and executes all 4 taggers one after the other)
    
```python rwtagger.py -conf -t direct indirect input_dir output_dir```

annotates only the types *direct* and *indirect*; outputs confidence values for each annotation; expects an input folder of plain text UTF-8 coded files 
        
 ```python rwtagger.py -gpu -f tsv input_dir output_dir``` 

runs the tagger on GPU; input format is not plain text but tsv (similar to the output format of the tagger: one token per line and markers for sentence start; column names must be 'tok' and 'sentstart'); annotates all 4 STWR types 


 ```python rwtagger.py -m test -t reported input_dir output_dir``` 

runs the tagger and also calculates test scores for the STWR type reported; input files must be tsv format and contain a column called 'reported' containing the gold standard annotations.


