# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 19:59:52 2020

@author: Safir Mohammad
"""


#import libraries
import os
import re
import sys
import nltk
import numpy as np
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

try:
    from nltk.corpus import stopwords
except:
    nltk.download('stopwords')
    from nltk.corpus import stopwords

#define class
class LSTMModel:  
    
    def __init__(self):
        pass
    
    
    #Method for importing dataset
    def importData(self, x_train, x_test):
        
        print("Importing Dataset... Please wait!")
        
        #Import testing data in x_test
        for filename in os.listdir("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\DataSet\\Test\\Negative\\"):
            with open(os.path.join("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\DataSet\\Test\\Negative\\", filename), 'r', encoding="utf8") as f:
                x_test.append(f.read())
      
        for filename in os.listdir("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\DataSet\\Test\\Positive\\"):
            with open(os.path.join("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\DataSet\\Test\\Positive\\", filename), 'r', encoding="utf8") as f:
                x_test.append(f.read())
      
        #Import training data in x_train
        for filename in os.listdir("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\DataSet\\Train\\Negative\\"):
            with open(os.path.join("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\DataSet\\Train\\Negative\\", filename), 'r', encoding="utf8") as f:
                x_train.append(f.read())
      
        for filename in os.listdir("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\DataSet\\Train\\Positive\\"):
            with open(os.path.join("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\DataSet\\Train\\Positive\\", filename), 'r', encoding="utf8") as f:
                x_train.append(f.read())

        print("Dataset successfully imported...!!!")
        return x_train, x_test
    
    
    #Method for pre-processing data
    def preProcessData(self, x_train, x_test):
        
        print("Pre-processing data... Please wait!")
        
        #Fetch all stopwords and keep required stopwords
        all_stopwords = stopwords.words('english')
        my_stopwords = [ word for word in all_stopwords if word not in ("against", "up", "down", "out", "off", "over", "under", "more", "most", "each", "few", "some", "such", "no", "nor", "not", "only", "too", "very", "don", "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't")]
        
        #Get rid of special characters
        REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
        REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
        
        for i in range(0,len(x_test)):
            #Keep only alphabets with single whitespace
            x_test[i] = REPLACE_NO_SPACE.sub("", x_test[i].lower())
            x_test[i] = REPLACE_WITH_SPACE.sub(" ", x_test[i])
            
            #Remove unwanted stopwords
            x_test[i] = x_test[i].split()
            x_test[i] = [ word for word in x_test[i] if word not in my_stopwords]
            x_test[i] = " ".join(x_test[i])
            
        for i in range(0,len(x_train)):
            #Keep only alphabets with single whitespace
            x_train[i] = REPLACE_NO_SPACE.sub("", x_train[i].lower())
            x_train[i] = REPLACE_WITH_SPACE.sub(" ", x_train[i])
            
            #Remove unwanted stopwords
            x_train[i] = x_train[i].split()
            x_train[i] = [ word for word in x_train[i] if word not in my_stopwords]
            x_train[i] = " ".join(x_train[i])
        
        
        print("Pre-processing done...!!!")
        return x_train, x_test
        
    
    #Method for Vectorizing(One-Hot) and Encoding data
    def encodeData(self, max_features, max_doc_len, x_train, x_test):
        
        print("Encoding data to One Hot Representation...")
        
        x_test = [ one_hot(document, max_features) for document in x_test]
        x_train = [ one_hot(document, max_features) for document in x_train]
        print(type(x_train))
        sc = StandardScaler()
        x_train = sc.fit_transform(np.array(x_train))
        x_test = sc.transform(np.array(x_test))
        
        #Add Bias
        for i in range(0, len(x_test)):
            x_test[i] = [1] + x_test[i]
        for i in range(0, len(x_train)):
            x_train[i] = [1] + x_train[i]
        
        #Word Embedding
        x_test = pad_sequences(x_test, truncating = 'post', padding = 'post', maxlen = max_doc_len)
        x_train = pad_sequences(x_train, truncating = 'post', padding = 'post', maxlen = max_doc_len)
        
        print("Encoding done...!!!")
        return x_train, x_test
    
    
    #Method for creating ML->RNN->LSTM model
    def createModel(self, max_features):
        
        print("Creating model...")
        model = MultinomialNB()
        
        print("Model created...!!!")
        return model
    
    
    #Method for training the model
    def trainModel(self, model, x_train, y_train, x_test, y_test):
        
        print("Training Model... Please wait!")
        model.fit(x_train, y_train)
        print("Model trained...!!!")
        return model
    
    def validateModel(self, model, x_test, y_test):
        y_pred = model.predict(x_test)
        print(accuracy_score(y_test, y_pred))
        
    #Method for predicting results through trained model
    def testModel(self, model, x_test, y_test):
        output = model.predict(x_test)
        print(output.shape)
        print(output)
        
        
        
    #Method for saving trained model
    def saveModel(self, model):
        #Serialize to JSON
        json_file = model.to_json()
        with open("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\Model_Bayes.json", "w") as file:
            file.write(json_file)
        
        #Serialize weights to HDF5
        model.save_weights("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\bayes_model_weights.h5")
        print("Model saved...")
        
    
    #Method for loading saved model
    def loadModel(self):
        #Load JSON and create model
        file = open("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\Model_Bayes.json", "r")
        model_json = file.read()
        file.close()
        
        loaded_model = model_from_json(model_json)
        #Load weights
        loaded_model.load_weights("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\bayes_model_weights.h5")
        print("Model loaded successfully...")
        
        return loaded_model
    
    
def main():
    try:
        #Declare train and test Variables
        x_train = []
        y_train = [0]*22750 + [1]*22750
        x_test = []
        y_test = [0]*2250 + [1]*2250
        #Define vocabulary size
        max_features = 25000
        #Define number of words per document/review
        max_doc_len = 80
        
        lstmModel = LSTMModel()
        x_train, x_test = lstmModel.importData(x_train, x_test)    
        x_train, x_test = lstmModel.preProcessData(x_train, x_test)
        x_train, x_test = lstmModel.encodeData(max_features, max_doc_len, x_train, x_test)
        model = lstmModel.createModel(max_features)
        model = lstmModel.trainModel(model, x_train, y_train, x_test, y_test)
        lstmModel.validateModel(model, x_test, y_test)
        lstmModel.testModel(model, x_test, y_test)
        
        #Save the trained model
        #lstmModel.saveModel(model)
        
        #Load the saved model
        #model = lstmModel.loadModel()
        
        #Try with new reviews
        new_neg_review = ["""A very charming film with wonderful sentiment and heart. It is rare when a film-maker takes the time to tell a worthy moral tale with care and love that doesn't fall into the trap of being overly syrupy or over indulgent. Nine out of ten for a truly lovely film.
                      """,
                      """I give this movie a 4 cause I'm a die hard fan of the video game series. the graphics and animation are excellent and its nice to see the whole gang in CG form Sephiroth's still cool<br /><br />now the reasons it only got a 4 well the characters feel like planks of wood with some of the worst voice acting I've ever seen(I've watched epic movie)<br /><br />the movie just seems cloud orientated so much so that it make even the fans embarrassed with cloud this and cloud that. clouds mentioned so much that it make you not want to see him in this movie <br /><br />the villains have the award for the worst villains ever (i was more scared by the wicked witch of the west) <br /><br />all the other characters in this movie are simply put in the movie for a nod to the fans and doesn't take it further then that<br /><br />wtf's with the chilly chally???<br /><br />summary: waited 9 years for this movie and this is what i get a large pointless and boring cut scene i beg the head of square cenix to shoot the man responsible for this burn every copy of the movie and any one involved in it and create a new movie from the ashes's (it would be nice to make the movie in live action and based on the original game)
                      """,
                      """The Movie is okay. Meaning that I don't regret watching it! I found the acting purely and the most of the dialog stupid ("oh no, this was my grandmothers bible!"). It's sort of bad remake of U-turn. A man arrives to a desert town out in nowhere, meets the wrong people and falls in love with the wrong woman. And off cause get's involved in something, he thought he could leave behind him. The movie is quite predictable and there is really nothing new in it. When it's finish, you didn't really care. Most of the characters are stereotypes, specially Brian Austin Green!! All in all just another movie from the states, but okay entertaining on boring Wednesday night. IMDb vote: 4/10
                      """,
                      """Hooray for Title Misspellings! After reading reviews and contemplating, my girlfriend and I confirmed that this movie is an utter piece of trash. This movie lost her as one of those Rare Tarantino fans.I wish it were made on nitrate film, and all the copies piled neatly underneath a chain-smoking Tarantino fanboy. The literally needless violence, the plot holes, Tarantino's table-itis sans drama, and absent character development made this a thoroughly painful, glorified montage. <br /><br />What acting was there? And how much of that was just because I was too busy reading the subtitles? I watch my share of fansubbed anime, and kudos for the attempt at authenticity, but it was overdone for an English-language movie. With the glaring historical inadequacies, the constant reading killed what acting there was.<br /><br />Why pay money for a narrator who will have absolutely no tie to any of the characters, plot, themes, setting, or anything involved in the movie? When the movie needs that sort of off hand explanation, it's foreshadowing the utter filth that follows.<br /><br />Historical Research - while it was sprinkled with interesting factiods, used the proper costumes and props for the soldiers, this movie stretched the truth beyond belief even for historical fiction. Kudos on Mata Hari reference, though using it as foreshadowing was a bit much. Mata Hari was executed by a firing squad, not choked in an isolated room. This ruined any sense that the reference may have had.<br /><br />Other reviews mention more than half a dozen homages to other artists in the first 15 minutes. Considering the audience, all these and other references were completely lost on many who would bother to see this movie, and all who would enjoy it.<br /><br />I'm confused by his choices of when to start a scene, end it, and what needs to be included. In a movie promoted as an action film, why did it take nearly 20 minutes to set up any sort of testosterone? <br /><br />What I believe to be the message was trite. The idea of rats and how we act on a primal nature against them, and "who is the rat?" were at best clichéd, but at worst not realized. Mention of American camps for the Japanese and German Americans would have added legitimacy to this question to the moral high ground. Literally every character in the film that gets a speaking role was caught up in their own legend. Is that the world in which Tarantino lives?<br /><br />I'm glad I didn't pay to see this one. I regret that I bothered to view it at all, even with well-meaning hosts. There was a rich base of ideas to develop, but none were realized.
                      """,
                      """Chapter One: Once Upon a Time
 At A Table (1941)<br /><br />In which a German Nazi and a French Dairy Farmer talk at a table for 20 minutes; first in French, then in English.<br /><br />Chapter Two: Three Years Of Inglorious Basterds In Sixteen Minutes... Without Tables (Mostly)<br /><br />In which an American Lieutenant talks to his newly formed 8 man Jewish- American commando unit. There are no tables present. Cut to Adolf Hitler, three years later. He is angry at his men's inability to deal with the Basterds. Hitler does have a table. We return to the Basterds in a flashback. Again, distinct lack of table-based content. <br /><br />Chapter Three: German Night in Paris... At A Table... Talking<br /><br />In which a Jewish woman who escaped from under the table in Chapter One has somehow managed to become the proprietress of a cinema. The Jewish woman talks to an Actor at a table in a bar. Later, the Jewish woman, the Actor, Joseph Goebbels and a Translator talk at a table in a Restaurant. The Actor and Goebbels talk in German. The Translator translates the German into French. The Jewish woman replies in French. The Translator translates the French into German. Goebbels decides to hold a film premiere at the Jewish woman's cinema. The Actor and Goebbels leave. The Nazi (who talked with the Dairy Farmer at a table for twenty minutes back in Chapter One) arrives. He talks with the Jewish woman at the table. He leaves. The Jewish woman breaks down; overcome with emotion at having spent so long talking at a table. <br /><br />Chapter Four: Operation Table Talking<br /><br />In which Austin Powers sends a British Officer to join the Basterds and an Actress on a mission to talk in German at a table in a Tavern. After 21 minutes of talking at a table they all shoot each other. The actress survives but spends the next 5 minutes lying on a table talking.<br /><br />Chapter Five: Revenge of the Giant Table<br /><br />In which, The Basterds decide to continue the operation by talking in Italian and suicide bombing the cinema. The Nazi takes the Actress into a small room where they sit next to a table. A hoe that he found under the table in the Tavern fits her so he kills her. Then he takes two of the Basterds to a big room, where they sit and talk at a table. Meanwhile, the cinema burns down, Hitler is riddled with bullets and the two Basterds blow themselves up for no good reason at all.<br /><br />The End
                      """]
        temp = []
        temp, new_neg_review = lstmModel.preProcessData(temp, new_neg_review)
        temp, new_neg_review = lstmModel.encodeData(max_features, max_doc_len, temp, new_neg_review)
        
        lstmModel.testModel(model, new_neg_review, temp)
        
        new_pos_review = ["""My boyfriend and I went to watch The Guardian.At first I didn't want to watch it, but I loved the movie- It was definitely the best movie I have seen in sometime.They portrayed the USCG very well, it really showed me what they do and I think they should really be appreciated more.Not only did it teach but it was a really good movie. The movie shows what the really do and how hard the job is.I think being a USCG would be challenging and very scary. It was a great movie all around. I would suggest this movie for anyone to see.The ending broke my heart but I know why he did it. The storyline was great I give it 2 thumbs up. I cried it was very emotional, I would give it a 20 if I could!
                          """,
                          """SPOILER ALERT! Don't read on unless you're prepared for some spoilers.<br /><br />I think this film had a lot beneath its shell. Besides the apparent connections with "Oldboy" (and Park-wook's other films), an incestuous relation in this one really disturbed me, and also the subtle erotic theme that hung around all the vampiric, physical action.<br /><br />The main actor, Kang-ho Song, is terrific in the rôle of the priest Sang-hyeon - coincidentally, "sang" means "blood" in some languages - who truly loved Tae-ju, played by OK-bin Kim. Their relationship reminds me a lot of that between Martin Sheen and Sissy Spacek in "Badlands", where the girl appears psychopathic and the man is basically wrapped around her finger.<br /><br />Their relationship is one thing, but the girl's mother is entirely different. While moving, she is stiff, one-dimensional and taut, but paralysed, she says all through not moving, or through the wink of an eye.<br /><br />Park-wook has really, really mastered his cinematography in this film, and owes a lot to Stanley Kubrick; there are a whole lot of beautiful shots strewn throughout the film, some for simple effects and some that require several glances and probably repeated views to fully catch.<br /><br />The music is quite stock, using mostly strings to accompany the main thespian's monoreaction; it's a very good thing that the character is as withdrawn as he is. While he does very little and loses at that, he seems to instead be a person who thinks a lot. While his love-interest says and does a lot, her actions display very little thought behind it. In my humble opinion.<br /><br />All in all, a very disturbing film that is not made for action, which isn't even in the same dimension as most things that are about vampires these days; it's magnificent, and repellant at the same time.
                          """,
                          """I know little or nothing about astronomy, but nevertheless; I was, at first, a little sceptical about the plot of this movie. It follows three children that were all born during a solar eclipse and so have no emotion, and thus (naturally) become ruthless serial killers. The plot does sound ridiculous at first, but once you realise that a solar eclipse blocks out Saturn and, as you know, Saturn is the emotion planet, it all falls into place; makes complete sense and it's then that you know you aren't simply watching another silly 80's slasher with a pea brain plot. Thank god for that! Seriously, though, Bloody Birthday is based on a ridiculous premise, but it more than makes up for that with it's originality. Having a bunch of kids going round slaughtering people may not be the most ingenious masterstroke ever seen in cinema, but when given the choice between this and another dull Friday the 13th clone - I know what I'd choose.<br /><br />Also helping the film out of the hole that some people would think it's silly plot dug it into is the fact that it's extremely entertaining. Many slashers become formulaic far too quickly and the audience ends up watching simply to see some gore. This film, however, keeps itself going with some great creepy performances from the kids (which harks back to creepy kid classics such as Village of the Damned), a constant stream of sick humour and a small, but impressive for the type of film, dose of suspense and tension. One thing that I liked a lot about this movie was the vast array of weaponry. There's nothing worse than a slasher where the killer uses the same weapon over and over again (cough Halloween cough), but that's not the case here as Bloody Birthday finds room for everything from skipping ropes to bow and arrows. There wasn't any room for a chainsaw, which is a huge shame, but I suppose not every film can have a chainsaw in it.
                          """,
                          """A very charming film with wonderful sentiment and heart. It is rare when a film-maker takes the time to tell a worthy moral tale with care and love that doesn't fall into the trap of being overly syrupy or over indulgent. Nine out of ten for a truly lovely film.
                          """,
                          """I absolutely adored this movie. For me, the best reason to see it is how stark a contrast it is from legal dramas like "Boston Legal" or "Ally McBeal" or even "LA Law." This is REALITY. The law is not BS, won in some closing argument or through some ridiculous defense you pull out of your butt, like the "Chewbacca defense" on South Park.) This is a real travesty of justice, the legal system gone horribly wrong, and the work by GOOD lawyers - not the shyster stereotype, who use all of their skills to right it. It will do more for restoring your faith in humanity than any Frank Capra movie or TO KILL A MOCKINGBIRD. And most importantly, I wept. During the film, during the featurette included at the end of the DVD - it's amazing. Wonderful film; wonderfully made. Thank God the filmmakers made it.
                          """]
    
        temp = []
        temp, new_pos_review = lstmModel.preProcessData(temp, new_pos_review)
        temp, new_pos_review = lstmModel.encodeData(max_features, max_doc_len, temp, new_pos_review)
        
        lstmModel.testModel(model, new_pos_review, temp)
        
    except:
        print("Unexpected error:", sys.exc_info())
    

if __name__ == "__main__":
    main()