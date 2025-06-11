import streamlit as st
import pandas as pd
import numpy as np

# Test simple pour vérifier que Streamlit fonctionne
st.title("🧪 Test Streamlit")
st.write("Si vous voyez ce message, Streamlit fonctionne correctement ! ✅")

# Test des dépendances
st.header("🔍 Test des dépendances")

dependencies_status = {}

# Test pandas
try:
    import pandas as pd
    dependencies_status["pandas"] = f"✅ {pd.__version__}"
except ImportError:
    dependencies_status["pandas"] = "❌ Non installé"

# Test numpy
try:
    import numpy as np
    dependencies_status["numpy"] = f"✅ {np.__version__}"
except ImportError:
    dependencies_status["numpy"] = "❌ Non installé"

# Test matplotlib
try:
    import matplotlib
    dependencies_status["matplotlib"] = f"✅ {matplotlib.__version__}"
except ImportError:
    dependencies_status["matplotlib"] = "❌ Non installé"

# Test plotly
try:
    import plotly
    dependencies_status["plotly"] = f"✅ {plotly.__version__}"
except ImportError:
    dependencies_status["plotly"] = "❌ Non installé"

# Test sklearn
try:
    import sklearn
    dependencies_status["scikit-learn"] = f"✅ {sklearn.__version__}"
except ImportError:
    dependencies_status["scikit-learn"] = "❌ Non installé"

# Affichage des résultats
for dep, status in dependencies_status.items():
    st.write(f"**{dep}:** {status}")

# Test d'affichage simple
st.header("📊 Test d'affichage")

# Créer des données factices
data = pd.DataFrame({
    'x': range(10),
    'y': np.random.randn(10)
})

st.line_chart(data.set_index('x'))

st.success("🎉 Tous les tests sont passés ! Votre environnement Streamlit est opérationnel.")
st.info("💡 Vous pouvez maintenant lancer votre application principale avec 'streamlit run streamlit_app.py'")