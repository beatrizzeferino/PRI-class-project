import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
#ingles 
from nltk.stem import PorterStemmer, WordNetLemmatizer 
#portugues
from nltk.stem.snowball import SnowballStemmer 
import spacy
import string

class TextProcessor:
    def __init__(self):

        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet= True)
        nltk.download('wordnet', quiet=True) 
        nltk.download('stopwords',quiet=True)
        nltk.download('omw-1.4', quiet=True)  

        # --- Recursos de INGLÊS ---
        self.stop_words_en = set(stopwords.words('english'))
        self.stemmer_en = PorterStemmer()
        self.lemmatizer_en = WordNetLemmatizer()

        # --- Recursos de PORTUGUÊS ---
        self.stop_words_pt = set(stopwords.words('portuguese'))
        self.stemmer_pt = SnowballStemmer('portuguese')

        # Carregamos o spaCy PT (md é um bom equilíbrio entre velocidade e precisão)
        try:
            self.nlp_pt = spacy.load("pt_core_news_md")
        except:
            print("[Aviso] Modelo spaCy PT não encontrado. Lematização PT pode falhar.")

        # Pontuação comum a ambos
        self.pontuacao = set(string.punctuation)
        self.pontuacao.update(['...', '«', '»', '“', '”', '’', '‘'])

    def segmentar_frases(self,text,language):
        """Divide o texto em frases."""
        return sent_tokenize(text, language=language)
    
    def tokenizar(self,frase,language):
        """Divide a frase em palavras"""
        return word_tokenize(frase, language=language)
    
    def get_wordnet_pos(self,tag):
        """Função auxiliar para mapear as tags POS para o formato do WordNet (Apenas Inglês)."""
        if tag.startswith('J'):
            w = wordnet.ADJ
        elif tag.startswith('V'): 
            w = wordnet.VERB 
        elif tag.startswith('N'): 
            w = wordnet.NOUN 
        elif tag.startswith('R'): 
            w = wordnet.ADV 
        else:          
            w = wordnet.NOUN   
        return w 

    def process_text(self, text, language='english', remove_stopwords=True, normalization_method=None):
        """
        Processa o texto com base nas configurações do sistema.
        - remove_stopwords (bool)
        - normalization_method (str): 'stem', 'lemma', ou None """

        # Colocar em minúsculas
        text = text.lower()
        
        #segmentacao de frases 
        frases = self.segmentar_frases(text, language)

        # Tokenização
        all_tokens = []
        for frase in frases:
            tokens_da_frase = self.tokenizar(frase, language)
            all_tokens.extend(tokens_da_frase)

        # remover a pontuação
        tokens = [word for word in all_tokens if word not in self.pontuacao]
        
        # Stop Words
        if remove_stopwords:
            sw = self.stop_words_pt if language == 'portuguese' else self.stop_words_en
            tokens = [word for word in tokens if word not in sw]
            
        # Normalização
        if normalization_method == 'stem':
            stemmer = self.stemmer_pt if language == 'portuguese' else self.stemmer_en
            tokens = [stemmer.stem(word) for word in tokens]
            
        elif normalization_method == 'lemma':
            if language == 'portuguese':
                # Lematização com spaCy PT
                doc = self.nlp_pt(" ".join(tokens))
                tokens = [token.lemma_ for token in doc]
            else:
                # Lematização com WordNet EN
                pos_tags = pos_tag(tokens)
                tokens = [self.lemmatizer_en.lemmatize(word, pos=self.get_wordnet_pos(tag)) 
                          for word, tag in pos_tags]
                
        return tokens
    
   
