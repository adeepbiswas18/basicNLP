import Utilities, os
import gensim

min_count = 1
context_window = 20
vector_size = 300
downsample = 1e-5
negative_sampling = 5
num_threads = 4
num_epochs = 10

#generates input tockens for the classifier
def getTrainTokens():
    trainContents = Utilities.getContents('train')
    trainTokens = Utilities.tokenizeContents(trainContents['Contents'])

    return {'Contents':trainContents['Contents'], 'Tokens':trainTokens, 'Labels':trainContents['Labels']}

#generates output label tockens for the classifier
def getTestTokens():
    testContents = Utilities.getContents('test')
    testTokens = Utilities.tokenizeContents(testContents['Contents'])

    return {'Contents':testContents['Contents'], 'Tokens':testTokens, 'Labels':testContents['Labels']}

#generates the Doc2Vec model from the tockens of the extual data
def trainDoc2Vec(tokens, savePath):
    docs = [gensim.models.doc2vec.TaggedDocument(words=token, tags=['DOC_' + str(idx)])
            for idx, token in enumerate(tokens)]

    if (os.path.exists(savePath)):
        model = gensim.models.Doc2Vec.load(savePath)
    else:
        model = gensim.models.Doc2Vec(docs, min_count=min_count, window=context_window, size=vector_size,
                                      sample=downsample, negative=negative_sampling, workers=num_threads,
                                      iter=num_epochs)
        model.save(savePath)

    return model