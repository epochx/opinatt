
Opinatt: Mining fine-grained opinions on closed captions of YouTube videos with an attention-RNN
==================

Code and dataset for the paper "Mining fine-grained opinions on closed captions of YouTube videos with an attention-RNN". If you use this code or dataset, please cite the paper.

1. Clone this repository ```cd ~; git clone https://github.com/epochx/opinatt``` 

2. Install required libraries
   * Install Theano and Tensorflow:  ```pip install Theano tensorflow-gpu```
   * Download and install Senna, http://ronan.collobert.com/senna/
   * Download and install CoreNLP 3.6, http://stanfordnlp.github.io/CoreNLP/history.html


1. Go to the opinatt diretory ```cd ~/opinatt``` and download the required data there
   * Download GoogleNews embeddings from https://code.google.com/archive/p/word2vec/
   * Download WikiDeps embeddings from https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/
   * Download and unzip the SemEval2014 V2 Train data from http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools

3. Create the working environment
    * Create data folder structure, ```prepare_data.sh path_where_to_create_data_folder```
    * Modify ```opinatt/enlp/settings.py``` accordingly

3. Preprocess corpora by running  ```python process_corpus.py```

4. Generate training data in JSON format: 
    * For aspect extraction run ```python generate_json.py --json_path path/to/json/file --train LaptopsTrain --test LaptopsTest --embeddings SennaEmbeddings```.
    * For aspect extraction and sentiment classification using collapsed tags, run ```python generate_json.py --json_path path/to/json/file --train LaptopsTrain --embeddings SennaEmbeddings --sentiment```. 
    *  For joint aspect extraction and sentiment classification run ```python generate_json.py --json_path path/to/json/file --train LaptopsTrain --embeddings SennaEmbeddings --joint```
    * When generating the training data for the Youtubean dataset, or when doing so including sentiment labels for the SemEval corpora, make sure not to pass the ```--test``` flag, as there are no test sets for those settings. The script will generate the corresponding splits based on the training portion for each case. Add the ```--strict``` label to only use sentences with a single sentiment. 



5. Train the models:
    * For the baseline: ```python run_baseline.py --json_path path/to/json/file --results_path path/to/results/folder```
    * For the attention-RNN: ```python run.py --json_path path/to/json/file --results_path path/to/results/folder```

For more details on how to use the provided scripts use the ```--help``` option.

**Contact**

Feel free to email emarrese@weblab.t.u-tokyo.ac.jp for any pertinent questions/bugs regarding the code. 
