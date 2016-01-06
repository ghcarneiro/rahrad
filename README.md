# rahrad
Adelaide University Summer Research Scholarship for Radiology at the Royal Adelaide Hospital

The data files are not included in the repository and need to be manually added.
The model files need to be generated before running the search engine

If you are running Linux you need to install the following packages:
  gfortran
  libfreetype6-dev
  libpng-dev
You may also want:
  liblapack-dev
  libblas-dev

The search engine requires the following python dependencies:
  numpy
  scipy
  cython
  statsmodels
  gensim
  nltk
  matplotlib
  sklearn
All of these can be installed with:
  easy_install --upgrade packageName

Once these are installed you need to download nltk's stopwords.
Open python and run:
  import nltk
  nltk.download()
The package identifier is "stopwords"
