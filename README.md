# rahrad
Adelaide University Summer Research Scholarship for Radiology at the Royal Adelaide Hospital

The model files were to large to be contained on github, this means that you need the Git Large File Storage (lfs) extension to work with them.
You can download this here:
https://git-lfs.github.com/
Once it is installed initialise it with:
  git lfs install
Then you need to download the large files with:
  git lfs fetch

The data files are not included in the repository and need to be manually added.

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
