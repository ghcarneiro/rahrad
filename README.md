# rahrad
Adelaide University Summer Research Scholarship for Radiology at the Royal Adelaide Hospital

The data files are not included in the repository and need to be manually added.
The model files need to be generated before running the search engine
The specialist lexicon dictionary needs to be build before any text preprocessing can be done
  This requires the LEXICON.xml file to be downloaded from:
    http://lsg3.nlm.nih.gov/LexSysGroup/Projects/lexicon/2016/release/LEX/XML/LEXICON.xml

If you are running Linux you need to install the following packages:
  gfortran
  libfreetype6-dev
  libpng-dev
  libyamal-dev
  libhdf5-dev
You may also want:
  liblapack-dev
  libopenblas-dev

The search engine requires the following python dependencies:
  numpy
  scipy
  cython
  statsmodels
  gensim
  nltk
  matplotlib
  sklearn
The RNN code also requires:
  pyyaml
  h5py
  keras (configured with theano backend)
  theano
All of these can be installed with:
  pip install --upgrade packageName

For setup of theano it is recommended that you make use of a gpu, for setup of Theano on ubuntu see:
  http://deeplearning.net/software/theano/install_ubuntu.html#install-ubuntu

Once these are installed you need to download nltk's stopwords.
Open python and run:
  import nltk
  nltk.download()
The package identifier is "stopwords"
