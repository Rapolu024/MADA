"""
Medical NLP Processor
Handles natural language processing for medical text data
"""

import re
import logging
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("Transformers not available. Using basic NLP processing.")

from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class MedicalNLPProcessor:
    """
    Processes medical text data for feature extraction and embedding generation
    """
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        
        # Medical terminology and mappings
        self.medical_synonyms = {
            'pain': ['ache', 'hurt', 'sore', 'tender', 'discomfort'],
            'nausea': ['sick', 'queasy', 'upset stomach'],
            'fatigue': ['tired', 'exhausted', 'weak', 'energy loss'],
            'fever': ['hot', 'temperature', 'burning up'],
            'headache': ['head pain', 'migraine'],
            'shortness of breath': ['difficulty breathing', 'breathless', 'winded'],
            'dizziness': ['lightheaded', 'vertigo', 'unsteady']
        }
        
        self.severity_indicators = {
            'mild': ['slight', 'little', 'minor', 'mild'],
            'moderate': ['moderate', 'noticeable', 'significant'],
            'severe': ['severe', 'intense', 'extreme', 'unbearable', 'terrible']
        }
        
        self.duration_patterns = {
            r'\b(\d+)\s*(day|days)\b': 'days',
            r'\b(\d+)\s*(week|weeks)\b': 'weeks',
            r'\b(\d+)\s*(month|months)\b': 'months',
            r'\b(\d+)\s*(year|years)\b': 'years',
            r'\b(\d+)\s*(hour|hours)\b': 'hours'
        }
        
        # Initialize models
        self._init_models()
    
    def _init_models(self):
        """Initialize NLP models and components"""
        # Initialize BERT model if transformers is available
        if HAS_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.eval()  # Set to evaluation mode
                logger.info(f"Loaded BERT model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load BERT model: {e}")
                HAS_TRANSFORMERS = False
        
        # Initialize TF-IDF vectorizer as fallback
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Medical vocabulary for TF-IDF
        self.medical_vocabulary = [
            'pain', 'nausea', 'fever', 'headache', 'fatigue', 'dizziness',
            'chest pain', 'abdominal pain', 'shortness breath', 'difficulty breathing',
            'heart rate', 'blood pressure', 'weight loss', 'weight gain'
        ]
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using BERT or TF-IDF"""
        if not text or text.strip() == "":
            return [0.0] * 768  # Return zero vector
        
        if HAS_TRANSFORMERS:
            return self._get_bert_embedding(text)
        else:
            return self._get_tfidf_embedding(text)
    
    def _get_bert_embedding(self, text: str) -> List[float]:
        """Generate BERT embedding for text"""
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                return embedding.tolist()
        
        except Exception as e:
            logger.error(f"Error generating BERT embedding: {e}")
            return [0.0] * 768
    
    def _get_tfidf_embedding(self, text: str) -> List[float]:
        """Generate TF-IDF embedding for text"""
        try:
            # Fit and transform if not already fitted
            if not hasattr(self.tfidf, 'vocabulary_'):
                # Use medical vocabulary to fit TF-IDF
                self.tfidf.fit(self.medical_vocabulary + [text])
            
            # Transform text
            tfidf_matrix = self.tfidf.transform([text])
            embedding = tfidf_matrix.toarray()[0].tolist()
            
            # Pad to 768 dimensions to match BERT
            if len(embedding) < 768:
                embedding.extend([0.0] * (768 - len(embedding)))
            else:
                embedding = embedding[:768]
            
            return embedding
        
        except Exception as e:
            logger.error(f"Error generating TF-IDF embedding: {e}")
            return [0.0] * 768
    
    def extract_symptoms(self, text: str) -> List[Dict[str, Any]]:
        """Extract symptoms and their characteristics from text"""
        symptoms = []
        text_lower = text.lower()
        
        # Common symptom patterns
        symptom_patterns = [
            r'\b(pain|ache|hurt|sore)\b',
            r'\b(nausea|sick|queasy)\b',
            r'\b(fever|temperature|hot)\b',
            r'\b(headache|head pain)\b',
            r'\b(fatigue|tired|exhausted)\b',
            r'\b(dizziness|lightheaded|vertigo)\b',
            r'\b(shortness of breath|difficulty breathing)\b',
            r'\b(cough|coughing)\b',
            r'\b(vomit|vomiting|throw up)\b'
        ]
        
        for pattern in symptom_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                symptom = {
                    'symptom': match.group(0),
                    'position': match.start(),
                    'severity': self._extract_severity(text, match.start()),
                    'duration': self._extract_duration(text)
                }
                symptoms.append(symptom)
        
        return symptoms
    
    def _extract_severity(self, text: str, position: int) -> Optional[str]:
        """Extract severity indicators near symptom mention"""
        # Look in a window around the symptom
        window_start = max(0, position - 50)
        window_end = min(len(text), position + 50)
        window_text = text[window_start:window_end].lower()
        
        for severity, indicators in self.severity_indicators.items():
            for indicator in indicators:
                if indicator in window_text:
                    return severity
        
        return None
    
    def _extract_duration(self, text: str) -> Optional[str]:
        """Extract duration information from text"""
        text_lower = text.lower()
        
        for pattern, unit in self.duration_patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                number = match.group(1)
                return f"{number} {unit}"
        
        return None
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of the text"""
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,  # -1 to 1
                'subjectivity': blob.sentiment.subjectivity  # 0 to 1
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from text"""
        entities = {
            'symptoms': [],
            'body_parts': [],
            'medications': [],
            'conditions': []
        }
        
        text_lower = text.lower()
        
        # Simple pattern matching for medical entities
        symptom_keywords = [
            'pain', 'ache', 'nausea', 'fever', 'headache', 'fatigue',
            'dizziness', 'cough', 'vomiting', 'diarrhea', 'constipation'
        ]
        
        body_part_keywords = [
            'head', 'chest', 'abdomen', 'stomach', 'back', 'leg', 'arm',
            'heart', 'lung', 'liver', 'kidney', 'throat', 'neck'
        ]
        
        condition_keywords = [
            'diabetes', 'hypertension', 'asthma', 'copd', 'depression',
            'anxiety', 'arthritis', 'migraine', 'infection'
        ]
        
        # Extract symptoms
        for keyword in symptom_keywords:
            if keyword in text_lower:
                entities['symptoms'].append(keyword)
        
        # Extract body parts
        for keyword in body_part_keywords:
            if keyword in text_lower:
                entities['body_parts'].append(keyword)
        
        # Extract conditions
        for keyword in condition_keywords:
            if keyword in text_lower:
                entities['conditions'].append(keyword)
        
        return entities
    
    def normalize_medical_text(self, text: str) -> str:
        """Normalize medical text using synonyms and standard terms"""
        normalized_text = text.lower()
        
        # Replace synonyms with standard terms
        for standard_term, synonyms in self.medical_synonyms.items():
            for synonym in synonyms:
                normalized_text = normalized_text.replace(synonym, standard_term)
        
        return normalized_text
    
    def calculate_text_complexity(self, text: str) -> Dict[str, Any]:
        """Calculate various text complexity metrics"""
        try:
            blob = TextBlob(text)
            sentences = blob.sentences
            words = blob.words
            
            # Basic metrics
            num_sentences = len(sentences)
            num_words = len(words)
            avg_words_per_sentence = num_words / num_sentences if num_sentences > 0 else 0
            
            # Character count
            num_characters = len(text)
            avg_chars_per_word = num_characters / num_words if num_words > 0 else 0
            
            return {
                'num_sentences': num_sentences,
                'num_words': num_words,
                'num_characters': num_characters,
                'avg_words_per_sentence': avg_words_per_sentence,
                'avg_chars_per_word': avg_chars_per_word
            }
        
        except Exception as e:
            logger.error(f"Error calculating text complexity: {e}")
            return {
                'num_sentences': 0,
                'num_words': 0,
                'num_characters': 0,
                'avg_words_per_sentence': 0,
                'avg_chars_per_word': 0
            }

