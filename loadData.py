import itertools
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.sparse
import pickle
import gc
from sklearn.feature_selection import VarianceThreshold

class DataLoader():
    def __init__(self, datasetName='static', dataPathSave='jkuLSCData/dataPythonReduced/dataPythonReduced/'):
        f=open(dataPathSave+'folds0.pckl', "rb")
        self.folds=pickle.load(f)
        f.close()

        f=open(dataPathSave+'labelsHard.pckl', "rb")
        targetMat=pickle.load(f)
        sampleAnnInd=pickle.load(f)
        targetAnnInd=pickle.load(f)
        f.close()

        targetMat=targetMat
        targetMat=targetMat.copy().tocsr()
        targetMat.sort_indices()
        targetAnnInd=targetAnnInd
        targetAnnInd=targetAnnInd-targetAnnInd.min()

        self.folds=[np.intersect1d(fold, sampleAnnInd.index.values).tolist() for fold in self.folds]
        targetMatTransposed=targetMat[sampleAnnInd[list(itertools.chain(*(self.folds)))]].T.tocsr()
        targetMatTransposed.sort_indices()
        trainPosOverall=np.array([np.sum(targetMatTransposed[x].data > 0.5) for x in range(targetMatTransposed.shape[0])])
        trainNegOverall=np.array([np.sum(targetMatTransposed[x].data < -0.5) for x in range(targetMatTransposed.shape[0])])



        self.denseOutputData=targetMat.A
        #self.denseOutputData=None
        self.sparseOutputData=targetMat



        if datasetName=="static":
          f=open(dataPathSave+'static.pckl', "rb")
          staticMat=pickle.load(f)
          sampleStaticInd=pickle.load(f)
          # featureStaticInd=pickle.load(f)
          f.close()
          
          self.denseInputData=staticMat
          self.denseSampleIndex=sampleStaticInd
          self.sparseInputData=None
          self.sparseSampleIndex=None
          
          del staticMat
          del sampleStaticInd
        elif datasetName=="semi":
          f=open(dataPathSave+'semi.pckl', "rb")
          semiMat=pickle.load(f)
          sampleSemiInd=pickle.load(f)
          # featureSemiInd=pickle.load(f)
          f.close()

          self.denseInputData=semiMat.A
          self.denseSampleIndex=sampleSemiInd
          self.sparseInputData=None
          self.sparseSampleIndex=None
          
          del semiMat
          del sampleSemiInd
        elif datasetName=="ecfp":
          f=open(dataPathSave+'ecfp6.pckl', "rb")
          ecfpMat=pickle.load(f)
          sampleECFPInd=pickle.load(f)
          # featureECFPInd=pickle.load(f)
          f.close()

          self.denseInputData=None
          self.denseSampleIndex=None
          self.sparseInputData=ecfpMat
          self.sparseSampleIndex=sampleECFPInd
          self.sparseInputData.eliminate_zeros()
          self.sparseInputData=sparseInputData.tocsr()
          self.sparseInputData.sort_indices()
          
          del ecfpMat
          del sampleECFPInd
          
          sparsenesThr=0.0025
        elif datasetName=="dfs":
          f=open(dataPathSave+'dfs8.pckl', "rb")
          dfsMat=pickle.load(f)
          sampleDFSInd=pickle.load(f)
          # featureDFSInd=pickle.load(f)
          f.close()

          self.denseInputData=None
          self.denseSampleIndex=None
          self.sparseInputData=dfsMat
          self.sparseSampleIndex=sampleDFSInd
          self.sparseInputData.eliminate_zeros()
          self.sparseInputData=sparseInputData.tocsr()
          self.sparseInputData.sort_indices()
          
          del dfsMat
          del sampleDFSInd
          
          sparsenesThr=0.02
        elif datasetName=="ecfpTox":
          f=open(dataPathSave+'ecfp6.pckl', "rb")
          ecfpMat=pickle.load(f)
          sampleECFPInd=pickle.load(f)
          # featureECFPInd=pickle.load(f)
          f.close()
          
          f=open(dataPathSave+'tox.pckl', "rb")
          toxMat=pickle.load(f)
          sampleToxInd=pickle.load(f)
          # featureToxInd=pickle.load(f)
          f.close()

          self.denseInputData=None
          self.denseSampleIndex=None
          self.sparseInputData=scipy.sparse.hstack([ecfpMat, toxMat])
          self.sparseSampleIndex=sampleECFPInd
          self.sparseInputData.eliminate_zeros()
          self.sparseInputData=sparseInputData.tocsr()
          self.sparseInputData.sort_indices()
          
          del ecfpMat
          del sampleECFPInd
          del toxMat
          del sampleToxInd
          
          sparsenesThr=0.0025

        gc.collect()



        allSamples=np.array([], dtype=np.int64)
        if not (self.denseInputData is None):
          allSamples=np.union1d(allSamples, self.denseSampleIndex.index.values)
        if not (self.sparseInputData is None):
          allSamples=np.union1d(allSamples, self.sparseSampleIndex.index.values)
        if not (self.denseInputData is None):
          allSamples=np.intersect1d(allSamples, self.denseSampleIndex.index.values)
        if not (self.sparseInputData is None):
          allSamples=np.intersect1d(allSamples, self.sparseSampleIndex.index.values)
        self.allSamples=allSamples.tolist()



        if not (self.denseInputData is None):
          self.folds=[np.intersect1d(fold, self.denseSampleIndex.index.values).tolist() for fold in self.folds]
        if not (self.sparseInputData is None):
          self.folds=[np.intersect1d(fold, self.sparseSampleIndex.index.values).tolist() for fold in self.folds]
