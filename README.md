# 🛒 Amazon Reviews - Analyse de Sentiments ⭐

## 📋 Description du Projet

Ce projet compare trois approches de machine learning pour prédire automatiquement la note (1-5 étoiles) d'une review Amazon à partir de son contenu textuel. L'objectif est d'évaluer l'efficacité du **Transfer Learning** avec les modèles Transformers par rapport à une approche baseline traditionnelle.

### 🎯 Objectifs
- Analyser les reviews Amazon Fine Food pour comprendre les patterns de sentiment
- Implémenter et comparer 3 modèles : Ridge Regression (baseline), BERT, et DistilBERT
- Appliquer des techniques de Transfer Learning sur des modèles pré-entraînés
- Développer une application Streamlit interactive pour la prédiction en temps réel

## 📊 Dataset

**Source** : [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) (Kaggle)

- **Taille** : ~500K reviews
- **Période** : Octobre 1999 - Octobre 2012
- **Variables principales** :
  - `Text` : Contenu textuel de la review
  - `Score` : Note de 1 à 5 étoiles
- **Preprocessing** : Nettoyage, suppression des stop words, tokenisation

### Justification du choix
- **Complexité** : Dataset large avec texte non-structuré
- **Défi NLP** : Analyse de sentiment multi-classe (régression)
- **Réalisme** : Cas d'usage industriel (e-commerce)

## 🏗️ Architecture du Projet

```
AmazonReviews-Regression/
├── .env                          # Environment variables
├── app/
│   ├── streamlit_app.py         # Main Streamlit application
│   └── test_streamlit.py        # Application tests
├── data/
│   └── Reviews.csv              # Raw dataset
├── notebooks/
│   ├── 0.2.6.0                  # Version directory
│   ├── 1_EDA_cleaning.ipynb     # Exploratory Data Analysis & Cleaning
│   ├── 2_Modeling_Training.ipynb # Model Development & Training
│   ├── clean_reviews.csv        # Cleaned dataset
│   ├── model_comparison.csv     # Model performance comparison
│   ├── random_forest_model.pkl  # Trained Random Forest model
│   ├── ridge_model.pkl          # Trained Ridge Regression model
│   ├── svr_model.pkl           # Trained Support Vector Regression model
│   └── tfidf_vectorizer.pkl    # TF-IDF vectorizer for text processing
├── .gitattributes              # Git LFS configuration
├── report.docx                 # Project report
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

## 🤖 Modèles Implémentés

### 1. 🔸 Ridge Regression (Baseline)
- **Architecture** : TF-IDF (10K features) + Ridge Regression
- **Justification** : Baseline rapide et interprétable
- **Avantages** : Temps d'entraînement < 1 min, faible complexité
- **Performance** : RMSE = 0.9335, MAE = 0.7022

### 2. 🔹 BERT (Transfer Learning)
- **Architecture** : BERT-base-uncased (110M paramètres)
- **Méthode Transfer Learning** :
  - Chargement du modèle pré-entraîné
  - Ajout d'une couche de régression (`num_labels=1`)
  - Fine-tuning complet avec AdamW optimizer
  - Learning rate scheduling avec warm-up
- **Justification** : État de l'art en NLP, représentations contextuelles bidirectionnelles
- **Performance** : RMSE = 0.8945, MAE = 0.6789

### 3. 🔹 DistilBERT (Transfer Learning Optimisé)
- **Architecture** : DistilBERT-base-uncased (66M paramètres)
- **Méthode Transfer Learning** : Identique à BERT
- **Justification** : 
  - Version "distillée" de BERT (40% plus petit)
  - Conserve 97% des performances de BERT
  - Meilleur compromis performance/efficacité
- **Performance** : RMSE = 0.8654, MAE = 0.6532 ⭐ **Meilleur modèle**

## 🚀 Installation et Utilisation

### Prérequis
```bash
Python 3.8+
CUDA compatible GPU (optionnel, pour accélération)
```

### Installation
```bash
# Cloner le repository
git clone https://github.com/[username]/amazon-reviews-sentiment.git
cd amazon-reviews-sentiment

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Lancement de l'application Streamlit
```bash
cd app
streamlit run streamlit_app.py
```

### Entraînement des modèles
```bash
# 1. Analyse exploratoire et nettoyage
jupyter notebook notebooks/EDA_cleaning.ipynb

# 2. Entraînement des modèles
jupyter notebook notebooks/Modeling_Training.ipynb
```

## 📈 Résultats

### Performance des Modèles

| Modèle | RMSE (Val) | MAE (Val) | Paramètres | Temps Training | Complexité |
|--------|------------|-----------|------------|----------------|------------|
| **DistilBERT** ⭐ | **0.8654** | **0.6532** | 66M | ~10 min | Élevée |
| BERT | 0.8945 | 0.6789 | 110M | ~15 min | Très Élevée |
| Ridge (Baseline) | 0.9335 | 0.7022 | 10K | < 1 min | Faible |

### Analyse des Résultats
- **🏆 Meilleur modèle** : DistilBERT (amélioration de 7.3% RMSE vs baseline)
- **⚡ Plus rapide** : Ridge Regression (excellent pour prototypage)
- **💡 Transfer Learning** : Amélioration significative grâce aux représentations pré-entraînées

## 🎯 Stratégie de Transfer Learning

### Méthodologie Appliquée
1. **Feature Extraction** : Utilisation des embeddings pré-entraînés
2. **Fine-tuning** : Adaptation des poids pour la tâche de régression
3. **Task-specific Head** : Ajout d'une couche dense pour la prédiction de score
4. **Optimisation** : AdamW avec learning rate scheduling

### Justification des Choix
- **BERT** : Référence en NLP, représentations contextuelles bidirectionnelles
- **DistilBERT** : Version optimisée, meilleur compromis performance/ressources
- **Régression** : `problem_type="regression"` avec MSE loss pour prédiction continue

## 🖥️ Application Streamlit

### Fonctionnalités
- **📊 Exploration des données** : Visualisations interactives, statistiques
- **🤖 Prédiction en temps réel** : Test des 3 modèles sur texte personnalisé
- **📈 Comparaison des modèles** : Métriques de performance, analyse
- **🔍 Analyse d'erreurs** : Identification des cas difficiles

### Interface
- Navigation par onglets
- Graphiques Plotly interactifs
- Prédictions comparatives en temps réel
- Exemples pré-définis pour test rapide

## 📊 Analyse Exploratoire (EDA)

### Insights Clés
- **Distribution déséquilibrée** : 40% de scores 5/5, 10% de scores 1/5
- **Corrélation longueur-sentiment** : Reviews positives généralement plus longues
- **Vocabulaire distinctif** : Mots-clés spécifiques par sentiment
- **Patterns temporels** : Évolution des sentiments sur la période

### Visualisations
- Distribution des scores (histogramme + camembert)
- Nuages de mots par sentiment
- Analyse de la longueur des textes
- Corrélations entre variables

## 🛠️ Technologies Utilisées

### Core ML/NLP
- **PyTorch** : Framework deep learning
- **Transformers (HuggingFace)** : Modèles pré-entraînés BERT/DistilBERT
- **Scikit-learn** : Modèle baseline, métriques, preprocessing
- **NLTK** : Traitement du langage naturel

### Visualisation & Interface
- **Streamlit** : Application web interactive
- **Plotly** : Graphiques interactifs
- **Matplotlib/Seaborn** : Visualisations statiques
- **WordCloud** : Nuages de mots

### Développement
- **Pandas/NumPy** : Manipulation de données
- **Jupyter** : Développement interactif
- **tqdm** : Barres de progression

## 📚 Méthodologie

### 1. Preprocessing
```python
# Nettoyage du texte
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Caractères spéciaux
    text = re.sub(r"\s+", " ", text)  # Espaces multiples
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)
```

### 2. Transfer Learning Implementation
```python
# Configuration BERT pour régression
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=1,  # Régression
    problem_type="regression"
)

# Fine-tuning avec AdamW
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, ...)
```

### 3. Métriques d'Évaluation
- **RMSE** : Erreur quadratique moyenne (pénalise les grandes erreurs)
- **MAE** : Erreur absolue moyenne (interprétable directement)
- **Validation croisée** : Split train/val/test (60%/20%/20%)

## 🔍 Analyse d'Erreurs

### Cas Difficiles Identifiés
- **Sarcasme** : "Great product... NOT!"
- **Contexte mixte** : Avis positif sur le produit, négatif sur la livraison
- **Nuances linguistiques** : Expressions idiomatiques
- **Reviews courtes** : Manque de contexte

## 🙏 Remerciements

- **Kaggle** pour le dataset Amazon Fine Food Reviews
- **HuggingFace** pour les modèles Transformers pré-entraînés
- **Streamlit** pour le framework d'application web
- Communauté open-source pour les outils utilisés

## 📞 Contact

Pour toute question sur ce projet :
- Email : adammhtazibert@gmail.com
- LinkedIn : https://www.linkedin.com/in/mahamat-azibert-adam
---

**⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile !**
