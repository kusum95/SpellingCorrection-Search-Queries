<a name="readme-top"></a>
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisite-packages">Prerequisite Packages</a></li>
        <li><a href="#running-the-script">Running the Script</a></li>
      </ul>
    </li>
    <li><a href="#preprocessed-datasets">Preprocessed Datasets</a></li>
    <li><a href="#choice-of-algorithm---methodology">Choice of Algorithm - Methodology</a>
      <ul>
        <li><a href="#language-model">Language Model</a></li>
        <li><a href="#main-method">Main Method</a></li>
        <li><a href="#spanish">Spanish</a></li>
      </ul></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>

# SpellingCorrection-Search-Queries
**Team members: Kusumakumari Vanteru, Mansi Tripathi, Nicole Santolalla, Yiqun Liu, Zhuyi(Elaine) Xu**

Git repo for the Spelling Correction model for Search 
Queries, developed by team *Spelling-Queens* as part of 
HackOverflow2023.

## Getting Started

Instructions on setting up the project locally.

### Prerequisite Packages

* **nltk**
  ```sh
  pip install nltk
  ```
* **enchant**
  ```sh
  brew install enchant
  pip install pyenchant
  
  pip install enchant (for Windows users)
  ```
  *If run into erros when installing, refer to this stackoverflow: <ins>https://stackoverflow.com/questions/29381919/importerror-the-enchant-c-library-was-not-found-please-install-it-via-your-o</ins>* <br />
  
* **lingua**
  ```sh
  pip install lingua-language-detector
  ```
* **googletrans**
  ```sh
  pip install googletrans==3.1.0a0
  ```
* Other more commonly used packages include: **numpy**, **csv**, **math**, **pandas**, **re**, **itertools**

### Running the Script

   ```sh
   python3 model_output_generation.py raw_query.csv
   ```
   
After running the above script, it will print out runtime and create a csv file called "**model_output.csv**". In the csv file, there are two columns, the first is "**raw_query**" and the second is "**corrected_query**", which suggest the corrections with words in lowercase and spaces normalized. If the input query is detected as Spanish, the "corrected_query" output will also be in Spanish.

**Note: If running into memory error, you can uncomment the ```gc.collector()``` at line 209 under _SpellCorrector.py_. The limitation being the runtime will increase drastically.**

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Preprocessed Datasets

In order for the model to select candidate words and calculate related probabilities, several files are preprocessed and prepared to serve as dictionaries for the model to train and learn:

* **custom_dict.txt** <br>
  - enchant package has preloaded dictionaries for English and Spanish, but after using those default dictionaries the performances were not good. So we developed customized dictionaries to include common dictionary words as well as words specific to Target (e.g. Target brand names). 
  - Detailed codes of generating this file is in compile_custom_dict.ipynb, it contains basic English words and brands Target sells (both in whole phrases and split into individual words). Our model will use this file as base dictionary to search for candidate correct words. If Target adds any brands, the brand names can be easily added to the txt file.

* **spanish_dict.txt** <br>
  - Similar to custom_dict.txt, this is the Spanish version of customized dictionary. The contents are derived by translating top Target search queries into Spanish and the model will use this file to search for candidate correct words when it is detecting Spanish inputs. 
  
* **big.txt** <br>
  - This dataset is from Peter Norvig and can be accessed from http://norvig.com/big.txt. It provides a large amount of corpus for the model to train.

* **brand_freq.csv** <br>
  - Queries to extract this dataset are in brand_freq.hql, which contains Target brands and their sales quantity in the past 2 years. The brand_freq file shows the weight for each brand so that top selling brands will have larger weights, and increases the model's performance in suggesting the most relevant correct words in the context of Target products.

* **top_search.xlsx** <br>
  - Queries to extract this dataset are in top_search.hql, which outputs top 100,000 corrected search queries generated by Target's search team in the past 3 years. We did minor manual corrections of the output file to ensure there are no spelling errors in it. Similar to brand_freq file, the top_search file will also help make the model suggestions more relevant.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Choice of Algorithm - Methodology

We use a noisy channel framework with fine tuned language models and dictionaries using Target data.

### Language Model

We train our languaage model on the Norvig corpus, Target search terms, and Target brand names. 

We **assigned weights** for search data and brand data frequencies so the count of occurences of a given word is $Norvig[word] + W_{search} * Search[word] + W_{brand} * Brand[word]$

### Main Method

Given input sentence " $word_1 word_2 word_3 ... word_n$ "

1. Get candidate words

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; For each word, we select **5** candidate words that might be the intended correct word using the dictionary and a python spellchecking <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; package _enchant_. <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; e.g. thee: three, the, there, thew, these

2. Get all possible sentences using different combinations of candidate words and raw input words

3. Calculate probabilities for each possible sentence and select one with the highest probability
    * Assume the probability of a word being a typo is **0.05** - this is to account for real word error where the inccrectedly spelled word is an actual English/Spanish word
    * Get the probability for each of the candidate word being the intended correct word given the raw input word using our error model - $P_{w_1}, P_{w_2}, ..., P_{w_n}$
    * Get bigram probabilities from our tuned language model for each of the bigrams in the sentence - $P_{b_1}, P_{b_2}, ..., P_{b_n}$
    * Multiply the probabilities together for each sentence - $P_{b_1} * ...  * P_{b_n} * P_{w_1} * ... * P_{w_n}$
    * Select the sentence with highest probability and output

### Spanish

1. Detect if the input phrase is Spanish or not using _lingua_ package
2. Repeat same steps as English using our self-defined Spanish Target dictionary and text file

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## References

1. Paper about noisy channel framework: <ins>https://web.stanford.edu/~jurafsky/slp3/B.pdf</ins>
2. Code reference: <ins>https://songyang0716.github.io/2017/06/an-implementation-of-noisy-channel-model/</ins>
3. Phonetic hashing and soundex: <ins>https://amitg0161.medium.com/phonetic-hashing-and-soundex-in-python-60d4ca7a2843</ins>
4. Python enchant package: <ins>https://pypi.org/project/pyenchant/</ins>
5. Enchant package install Q&A: <ins>https://stackoverflow.com/questions/29381919/importerror-the-enchant-c-library-was-not-found-please-install-it-via-your-o</ins>
6. Enchant customize dictionary: <ins>https://we-are.bookmyshow.com/building-a-movies-dictionary-spell-checker-7dd6f6d897ff</ins>

<p align="right">(<a href="#readme-top">back to top</a>)</p>
