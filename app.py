import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc

st.set_page_config(page_title="Detección de anomalias", page_icon="📊", layout="wide")

st.markdown("""
<div style='font-size:42px; text-align: center; font-weight: bold'>
Detección de fraudes financieros
""", unsafe_allow_html=True)

st.markdown("""
<div style='font-size:18px; text-align: justify'>
En la entidad bancaria internacional <strong>TioRicoBank</strong>, que procesa diariamente más de cinco millones de transacciones financieras. 
En este entorno de alta presión y regulación, tu equipo ha sido comisionado para diseñar un sistema de detección temprana de fraudes 
financieros con las siguientes condiciones estrictas:
</div>
            
<ul style="list-style-type: disc; padding-left: 20px; margin-top: 10px;">
    <li style="font-size:18px;"><strong>Tiempo de respuesta:</strong> menor a <strong>100 ms</strong> por transacción.</li>
    <li style="font-size:18px;"><strong>Sensibilidad extrema ante fraudes:</strong> se exige un <strong>Recall mayor al 90 %</strong> en la clase minoritaria.</li>
    <li style="font-size:18px;"><strong>Alta precisión:</strong> menos del <strong> 1% de falsos positivos</strong>.</li>
    <li style="font-size:18px;"><strong>Explicabilidad jurídica:</strong> cada transacción marcada como sospechosa debe poder justificarse de manera clara ante auditores y entes legales.</li>
</ul>

<div style='font-size:18px; text-align: justify'>
Para ello se cuenta con un histórico de transacciones anonimizado que incluye 284.807 registros, de los cuales solo 492 corresponden a fraudes
reales. Las variables V1 a V28 fueron generadas mediante una descomposición PCA para proteger la privacidad de los datos originales.
Adicionalmente se incluyen las variables Time y Amount que son relevantes para determinar el comportamiento de las transacciones como se puede ver acontinuación:
</div>
<br>
""", unsafe_allow_html=True)

if "df" not in st.session_state:
    df = pd.read_csv("https://zenodo.org/records/7395559/files/creditcard.csv")
    st.session_state.df = df
else:
    df = st.session_state.df

st.dataframe(df.head(),hide_index=True)

df.sort_values("Class",inplace=True)

X_train = df.iloc[:283823].drop(columns=['Class'])
X_test = df.iloc[283823:].drop(columns=['Class'])
y_test = df.iloc[283823:]['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

st.markdown(f"""
<div style='text-align: center; font-size: 32px; font-weight: bold'>            
PCA para deteccion de anomalias
</div>
                        
<div style='font-size:18px; text-align: justify'>
Para usar el PCA como metodo de deteccion de anomalias lo primero que haremos es aplicar PCA a los datos de entrenamiento donde solo tendremos datos
que no fueron fraudes para que los componentes sean lo mas representativos posibles de los datos normales. Seleccionamos el número óptimo de componentes 
que expliquen al menos el 95% de la varianza, que se consigue con {round(pca.n_components_, 2)} componentes principales.
</div>
""", unsafe_allow_html=True)

with st.expander("Ver código"):
    st.code("""
import pandas as pd
import numpy as np 
                       
df = pd.read_csv("creditcard.csv")                      ### Cargar los datos
df.sort_values("Class", inplace=True)                   ### Ordenar los datos por la clase (fraude vs no fraude)

X_train = df.iloc[:283823].drop(columns=['Class'])      ### Tomar los datos de entrenamiento (sin fraudes)
X_test = df.iloc[283823:].drop(columns=['Class'])       ### Tomar los datos de prueba (50% fraudes y 50% no fraudes)
y_test = df.iloc[283823:]['Class']                      ### Etiquetas de los datos de prueba

scaler = StandardScaler()                               
X_scaled = scaler.fit_transform(X_train)                ### Normalizar los datos de entrenamiento 

pca = PCA(n_components=0.95)                            ### Seleccionar el número óptimo de componentes que expliquen al menos el 95% de la varianza
X_pca = pca.fit_transform(X_scaled)                     ### Aplicar PCA a los datos de entrenamiento
""", language='python')

st.markdown("""
<div style='font-size:18px; text-align: justify'>
Calcula para cada transacción su distancia de reconstrucción en el subespacio PCA, y utiliza esta métrica como indicador de anomalía. 
Compara su distribución entre clases (fraude vs. no fraude), y construye la curva ROC respectiva.
</div>""", unsafe_allow_html=True)

with st.expander("Ver código"):
    st.code("""
X_test_scaled = scaler.transform(X_test)                    ### Normalizar los datos de prueba  
X_pca_test = pca.transform(X_test_scaled)                   ### Aplicar PCA a los datos de prueba
X_test_reconstructed = pca.inverse_transform(X_pca_test)    ### Reconstruir los datos de prueba a partir de los componentes principales

### Error de reconstrucción promedio
reconstruction_error_test = np.mean((X_test_scaled - X_test_reconstructed) ** 2, axis=1)            
""", language='python')
    
X_test_scaled = scaler.transform(X_test)
X_pca_test = pca.transform(X_test_scaled)
X_test_reconstructed = pca.inverse_transform(X_pca_test)

reconstruction_error_test = np.mean((X_test_scaled - X_test_reconstructed) ** 2, axis=1)

df_pca_test = pd.DataFrame({
    "reconstruction_error": reconstruction_error_test,
    "Class": y_test.values
})

fpr, tpr, thresholds = roc_curve(df_pca_test["Class"], df_pca_test["reconstruction_error"])
roc_auc = auc(fpr, tpr)

umbral2 = df_pca_test[df_pca_test["Class"] == 0]["reconstruction_error"].quantile(0.95)
tn, fp, fn, tp = confusion_matrix(y_test, (reconstruction_error_test > umbral2).astype(int)).ravel()

tpr2 = tp / (tp + fn)
fpr2 = fp / (fp + tn)

f1_scores = 2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr))
best_f1_idx = f1_scores.argmax()
best_threshold_f1 = thresholds[best_f1_idx]
best_f1_score = f1_scores[best_f1_idx]

y_pred = (reconstruction_error_test > best_threshold_f1).astype(int)

cm = confusion_matrix(y_test, y_pred)
reporte = classification_report(y_test, y_pred, output_dict=True, digits=4)

df_reporte = pd.DataFrame(reporte).drop(index=["support"])

df_reporte = df_reporte.drop(columns=["accuracy", "macro avg", "weighted avg"])

df_reporte = df_reporte.transpose()

st.markdown("""
<div style='text-align: center; font-size: 28px; font-weight: bold'>
Análisis de Reconstrucción PCA
</div>
""", unsafe_allow_html=True)

# Crear subplots: 1 fila, 2 columnas
fig = make_subplots(rows=1, cols=2, 
                    subplot_titles=("Error de reconstrucción por clase", "Curva ROC"),
                    column_widths=[0.45, 0.55])

# Boxplot (col 1)
fig.add_trace(go.Box(
    y=df_pca_test[df_pca_test["Class"] == 0]["reconstruction_error"],
    name="No Fraude (0)", showlegend=False, marker_color='lightblue', boxmean='sd'
), row=1, col=1)

fig.add_trace(go.Box(
    y=df_pca_test[df_pca_test["Class"] == 1]["reconstruction_error"],
    name="Fraude (1)", showlegend=False, marker_color='tomato', boxmean='sd'
), row=1, col=1)

# ROC Curve (col 2)
fig.add_trace(go.Scatter(
    x=fpr, y=tpr,
    mode='lines',
    showlegend=False,
    name=f'ROC curve (AUC = {roc_auc:.4f})',
    line=dict(color='darkorange', width=3)
), row=1, col=2)

fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode='lines',
    showlegend=False,
    line=dict(color='gray', dash='dash')
), row=1, col=2)

# Agregar punto del mejor umbral
fig.add_trace(go.Scatter(
    x=[fpr[best_f1_idx]],
    y=[tpr[best_f1_idx]],
    mode='markers+text',
    marker=dict(color='red', size=10, symbol='circle'),
    hovertext=[f"<b>Umbral óptimo (F1):</b> {best_threshold_f1:.4f}"],
    hoverinfo='text',
    name = "Umbral F1"
), row=1, col=2)

# Agregar punto del mejor umbral
fig.add_trace(go.Scatter(
    x=[fpr2],
    y=[tpr2],
    mode='markers+text',
    marker=dict(color='blue', size=10, symbol='circle'),
    hovertext=[f"<b>Umbral percentil 95:</b> {umbral2:.4f}"],
    name = "Umbral percentil 95",
    hoverinfo='text'
), row=1, col=2)

# Layout general
fig.update_layout(
    legend=dict(
        orientation="h",       # Horizontal
        yanchor="bottom",      # Anclado a la parte inferior
        y=-0.3,                # Un poco por debajo de la figura
        xanchor="center",      
        x=0.75                  # Centrado horizontalmente
    ),
    height=450,
    width=1000,
    margin=dict(l=40, r=40, t=60, b=40)
)

fig.update_yaxes(title_text="Error de reconstrucción", row=1, col=1)
fig.update_xaxes(title_text="FPR", row=1, col=2)
fig.update_yaxes(title_text="TPR", row=1, col=2)

# Mostrar en Streamlit
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div style='text-align: center; font-size: 28px; font-weight: bold'>
Reporte de Clasificación
</div>""", unsafe_allow_html=True)

col1, _ , col2 = st.columns([0.6, 0.1, 0.6])

with col1:
    st.markdown("""
                <div style='text-align: center; font-size: 20px'>
                <b>Rendimiento con umbral optimo para F1</b>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("Matriz de Confusión")
    st.dataframe(pd.DataFrame(cm, index=["No Fraude (0)", "Fraude (1)"], columns=["No Fraude (0)", "Fraude (1)"]), use_container_width=True)
    st.markdown("Metricas")
    st.dataframe(df_reporte.style.format(precision=4), use_container_width=True)

with col2:
    y_pred = (reconstruction_error_test > umbral2).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    reporte = classification_report(y_test, y_pred, output_dict=True, digits=4)

    df_reporte = pd.DataFrame(reporte).drop(index=["support"])
    df_reporte = df_reporte.drop(columns=["accuracy", "macro avg", "weighted avg"])
    df_reporte = df_reporte.transpose()

    st.markdown("""
                <div style='text-align: center; font-size: 20px'>
                <b>Rendimiento con umbral del percentil 95</b>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("Matriz de Confusión")
    st.dataframe(pd.DataFrame(cm, index=["No Fraude (0)", "Fraude (1)"], columns=["No Fraude (0)", "Fraude (1)"]), use_container_width=True)
    st.markdown("Metricas")
    st.dataframe(df_reporte.style.format(precision=4), use_container_width=True)

st.markdown("""
<div style='font-size:18px; text-align: justify'>
Usar PCA en lugar de otros metodos como Autoencoders deja varias preguntas, por ejemplo ¿Cómo se comporta este método respecto a los Autoencoders en 
términos de capacidad discriminativa, facilidad de implementación y potencial de explicabilidad? En el caso de los Autoencoders, su capacidad de aprender
suelen tener mayor capacidad discriminativa, especialmente en datos complejos y no lineales, pero por otro lado requieren arquitecturas más complejas y
más pasos como lo son el diseño de red, entrenamiento, ajuste de hiperparámetros para alcanzar un rendimiento óptimo, que puede ser un desafío en términos de
tiempo y recursos computacionales si no se cuenta con una GPU.
</div>
<br>
<div style='font-size:18px; text-align: justify'> 
Por otro lado, el PCA es un método Lineal, lo que lo limita frente a estructuras complejas en los datos, pero como estos datos son componentes de PCA, ya tiene
relaciones lineales entre ellos, lo que hace que el PCA sea una buena opción. Además, el PCA es más fácil de implementar y no requiere un proceso de entrenamiento 
complejo, pero para este caso en particular, tenemos un enorme pero, y es que estas variables ya vienen de un PCA, por lo que aplicar PCA sobre un conjunto de 
variables que ya corresponden a componentes principales obtenidas previamente no introduce nueva información ni contribuye a una simplificación real del conjunto de 
datos. En términos matemáticos, esta segunda aplicación de PCA resulta en una proyección sobre una base ortonormal que coincide con la base canónica, lo cual implica 
que la transformación no altera las representaciones originales, osea que el espacio latente permanece esencialmente invariante. Esto coinside con que se necesiten de
26 componentes para alcanzar el 95% de la varianza, solo 2 componentes menos que el PCA original. Todo esto hace que el PCA tenga sus pequeños peros tecnicos que pueden
afectar en la calidad del modelo, pero en general es un buen metodo para detectar anomalias si se tiene relaciones lineales en los datos y se necesita de poco computo.
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style='text-align: center; font-size: 32px; font-weight: bold'>            
Diseño de Autoencoder
</div>
<div style='font-size:18px; text-align: justify'>
Antes de entrenar un Autoencoder es fundamental hacer un preprocesamiento previo de las variables de entrada, especialmente aquellas que presentan una alta dispersión 
o valores atípicos extremos. En este caso, se sugiere utilizar RobustScaler para escalar la variable Amount en lugar de un escalador como StandardScaler. La razón 
principal es que RobustScaler realiza el escalado basado en la mediana y el rango intercuartílico (IQR), en lugar de la media y la desviación estándar. Esto lo hace 
mucho más resistente a la influencia de outliers, que son frecuentes en variables económicas como el monto de una transacción.
</div>
<br>
<div style='font-size:18px; text-align: justify'>
En este contexto, un escalado sensible a valores atípicos (como el que realiza StandardScaler) puede distorsionar la estructura interna de los datos. Los outliers 
podrían dominar el proceso de aprendizaje, forzando al modelo a enfocarse en reconstruir valores extremos en lugar de capturar la estructura de los datos. Esto 
deterioraría su capacidad de detectar patrones de anomalía, ya que el Autoencoder debería reconstruir bien las transacciones no fraudulentas y fallar en las 
transacciones fraudulentas.
</div>
<br>
""", unsafe_allow_html=True)

scaler = RobustScaler()
X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])
X_test['Amount'] = scaler.transform(X_test[['Amount']])

with st.expander("Ver código"):
    st.code("""
scaler = RobustScaler()                                         ### Escalador robusto para manejar outliers
X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])   ### Escalar la variable Amount en el conjunto de entrenamiento
X_test['Amount'] = scaler.transform(X_test[['Amount']])         ### Escalar la variable Amount en el conjunto de prueba con parametros de entrenamiento
""", language='python')

col1, col2 = st.columns(2)

with col1:
    if 'fig_boxplot' not in st.session_state:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Class', y='Amount', data=df, ax=ax)
        ax.set_title('Distribución del Monto por Tipo de Transacción')
        ax.set_xlabel('Clase')
        ax.set_ylabel('Monto')
        ax.set_yscale('log')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Legítima', 'Fraudulenta'])
        st.session_state.fig_boxplot = fig

    # Mostrar la figura desde el session_state
    st.pyplot(st.session_state.fig_boxplot)

with col2:
    scaler = RobustScaler()
    df['Amount2'] = scaler.fit_transform(df[['Amount']])
    
    if 'fig_boxplot_scaled' not in st.session_state:    
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Class', y='Amount2', data=df, ax=ax)
        ax.set_title('Distribución del Monto escalado')
        ax.set_xlabel('Clase')
        ax.set_ylabel('Monto')
        ax.set_yscale('log')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Legítima', 'Fraudulenta'])
        st.session_state.fig_boxplot_scaled = fig

    # Mostrar la figura guardada
    st.pyplot(st.session_state.fig_boxplot_scaled)

st.markdown("""
<div style='font-size:18px; text-align: justify'>
La comparación de las distribuciones de la variable Amount antes y después de aplicar RobustScaler muestra que el escalado basado en la mediana y el rango 
intercuartílico logra comprimir la dispersión de los datos, reduciendo drásticamente la influencia de los valores atípicos. Mientras que en el estado original 
los montos presentan una alta varianza y una gran cantidad de outliers que dominan la escala, después del escalado robusto los datos quedan centrados en torno a 
valores más representativos de la mayoría de las transacciones, sin eliminar los outliers pero limitando su impacto.
</div>
<br>
<div style='font-size:18px; text-align: justify'>
Las tres arquitecturas de Autoencoder simétrico propuestas son AE1 (30-20-10-20-30), AE2 (30-25-15-25-30) y AE3 (30-28-26-28-30). presentan diferentes configuraciones 
de tamaños en el cuello de botella, lo que influirá en su capacidad de compresión, generalización y propensión al sobreajuste. AE1, con un cuello de botella de 10 neuronas, 
tiene una mayor capacidad de compresión, lo que puede llevar a una mayor pérdida de información relevante, lo que puede llevar a aumentar el error de reconstrucción, pero 
también reduciendo el riesgo de sobreajuste. AE2, con un tamaño intermedio en el cuello de botella de 15 neuronas, ofrece un balance entre compresión y capacidad para 
aprender la estructura de los datos, lo que podría mejorar la generalización a nuevos datos sin un exceso de sobreajuste. AE3, con un cuello de botella más
grande de 26 neuronas que esta mas cerca a las 30 de variables de entrada, esto hace que se preserven más detalles, lo que puede reducir el error de reconstrucción, 
pero también incrementa el riesgo de sobreajuste al aprender representaciones muy específicas de los datos de entrenamiento.
</div>
<br>
<div style='font-size:18px; text-align: justify'>
A continuación se presentan las tres arquitecturas de Autoencoder propuestas y su respectivo código de implementación y su entrenamiento.
</div>
""", unsafe_allow_html=True)

with st.expander("Ver código"):
    st.code("""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Clase base para facilitar reutilización
class BaseAutoencoder(nn.Module):
    def __init__(self):
        super(BaseAutoencoder, self).__init__()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

### Autoencoder 1 (30-20-10-20-30)
class AE1(BaseAutoencoder):
    def __init__(self):
        super(AE1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30)
        )

### Autoencoder 2 (30-25-15-25-30)
class AE2(BaseAutoencoder):
    def __init__(self):
        super(AE2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(30, 25),
            nn.ReLU(),
            nn.Linear(25, 15),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(15, 25),
            nn.ReLU(),
            nn.Linear(25, 30)
        )
            
### Autoencoder 3 (30-28-26-28-30)
class AE3(BaseAutoencoder):
    def __init__(self):
        super(AE3, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(30, 28),
            nn.ReLU(),
            nn.Linear(28, 26),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(26, 28),
            nn.ReLU(),
            nn.Linear(28, 30)
        )

from torch.utils.data import DataLoader, TensorDataset

# Asegura reproducibilidad
torch.manual_seed(42)

# Si hay GPU, úsala
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convierte los datos en tensores
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)

# DataLoaders
train_loader = DataLoader(TensorDataset(X_train_tensor, X_train_tensor), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, X_test_tensor), batch_size=64, shuffle=False)

def train_autoencoder(model, train_loader, test_loader, epochs=50, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x_batch, _ in train_loader:
            output = model(x_batch)
            loss = criterion(output, x_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))
        
        # Evaluación
        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(xb), xb).item() for xb, _ in test_loader) / len(test_loader)
        test_losses.append(val_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d}: Train Loss = {train_losses[-1]:.4f}, Test Loss = {test_losses[-1]:.4f}")
    
    return train_losses, test_losses

### Instanciar los modelos 
ae1 = AE1()
ae2 = AE2()
ae3 = AE3()

### Usar funcion creada para entrenar los modelos
loss_ae1, val_ae1 = train_autoencoder(ae1, train_loader, test_loader)
loss_ae2, val_ae2 = train_autoencoder(ae2, train_loader, test_loader)
loss_ae3, val_ae3 = train_autoencoder(ae3, train_loader, test_loader)
""", language='python')
    
# Clase base para facilitar reutilización
class BaseAutoencoder(nn.Module):
    def __init__(self):
        super(BaseAutoencoder, self).__init__()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

### Autoencoder 1 (30-20-10-20-30)
class AE1(BaseAutoencoder):
    def __init__(self):
        super(AE1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30)
        )

### Autoencoder 2 (30-25-15-25-30)
class AE2(BaseAutoencoder):
    def __init__(self):
        super(AE2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(30, 25),
            nn.ReLU(),
            nn.Linear(25, 15),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(15, 25),
            nn.ReLU(),
            nn.Linear(25, 30)
        )
            
### Autoencoder 3 (30-28-26-28-30)
class AE3(BaseAutoencoder):
    def __init__(self):
        super(AE3, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(30, 28),
            nn.ReLU(),
            nn.Linear(28, 26),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(26, 28),
            nn.ReLU(),
            nn.Linear(28, 30)
        )

from torch.utils.data import DataLoader, TensorDataset

# Asegura reproducibilidad
torch.manual_seed(42)

# Si hay GPU, úsala
device = torch.device("cpu")

# Convierte los datos en tensores
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)

# DataLoaders
train_loader = DataLoader(TensorDataset(X_train_tensor, X_train_tensor), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, X_test_tensor), batch_size=64, shuffle=False)

ae1 = AE1()
ae1.load_state_dict(torch.load('ae1.pth',map_location=device))
ae1.to(device)

ae2 = AE2()
ae2.load_state_dict(torch.load('ae2.pth',map_location=device))
ae2.to(device)

ae3 = AE3()
ae3.load_state_dict(torch.load('ae3.pth',map_location=device))
ae3.to(device)

def evaluate_autoencoder(model, X_test_tensor, X_test_df, y_test):
    model.eval()
    with torch.no_grad():
        reconstructions = model(X_test_tensor).cpu().numpy()
    
    errors = np.mean((reconstructions - X_test_df.values)**2, axis=1)
    threshold = np.percentile(errors[y_test == 0], 95)
    y_pred = (errors > threshold).astype(int)

    return errors

def plot_reconstruction_error_and_roc(errors, y_test):
    # Crear DataFrame para análisis
    df_errors = pd.DataFrame({
        "reconstruction_error": errors,
        "Class": y_test.values
    })

    # Calcular curva ROC y AUC
    fpr, tpr, thresholds = roc_curve(df_errors["Class"], df_errors["reconstruction_error"])
    roc_auc = auc(fpr, tpr)

    # Crear subplots: 1 fila, 2 columnas
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=("Error de reconstrucción por clase", "Curva ROC (AUC = {:.4f})".format(roc_auc)),
                        column_widths=[0.45, 0.55])

    # Boxplot (col 1)
    fig.add_trace(go.Box(
        y=df_errors[df_errors["Class"] == 0]["reconstruction_error"],
        name="No Fraude (0)", showlegend=False, marker_color='lightblue', boxmean='sd'
    ), row=1, col=1)

    fig.add_trace(go.Box(
        y=df_errors[df_errors["Class"] == 1]["reconstruction_error"],
        name="Fraude (1)", showlegend=False, marker_color='tomato', boxmean='sd'
    ), row=1, col=1)

    # ROC Curve (col 2)
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        showlegend=False,
        name=f'ROC curve (AUC = {roc_auc:.4f})',
        line=dict(color='darkorange', width=3)
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        showlegend=False,
        line=dict(color='gray', dash='dash')
    ), row=1, col=2)

    # Layout general
    fig.update_layout(
        legend=dict(
            orientation="h",       # Horizontal
            yanchor="bottom",      # Anclado a la parte inferior
            y=-0.3,                # Un poco por debajo de la figura
            xanchor="center",      
            x=0.5                  # Centrado horizontalmente
        ),
        height=450,
        width=1000,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    fig.update_yaxes(title_text="Error de reconstrucción", row=1, col=1)
    fig.update_xaxes(title_text="FPR", row=1, col=2)
    fig.update_yaxes(title_text="TPR", row=1, col=2)

    return fig

st.write(f"X_train shape: {X_train.shape}")
st.write(f"X_test shape: {X_test.shape}")

st.write(X_test_tensor.shape)

errors_ae1 = evaluate_autoencoder(ae1, X_test_tensor, X_test, y_test)
errors_ae2 = evaluate_autoencoder(ae2, X_test_tensor, X_test, y_test)
errors_ae3 = evaluate_autoencoder(ae3, X_test_tensor, X_test, y_test)

# Llamar a la función para cada modelo
fig_ae1 = plot_reconstruction_error_and_roc(errors_ae1, y_test)
fig_ae2 = plot_reconstruction_error_and_roc(errors_ae2, y_test)
fig_ae3 = plot_reconstruction_error_and_roc(errors_ae3, y_test)

st.markdown("""
<div style='text-align: center; font-size: 28px; font-weight: bold'>
Analisis de reconstrucción AE1
</div>
""", unsafe_allow_html=True)
st.plotly_chart(fig_ae1, use_container_width=True)

st.markdown("""
<div style='text-align: center; font-size: 28px; font-weight: bold'>
Analisis de reconstrucción AE2
</div>
""", unsafe_allow_html=True)
st.plotly_chart(fig_ae2, use_container_width=True)

st.markdown("""
<div style='text-align: center; font-size: 28px; font-weight: bold'>
Analisis de reconstrucción AE3
</div>
""", unsafe_allow_html=True)
st.plotly_chart(fig_ae3, use_container_width=True)

st.markdown("""
<div style='font-size:18px; text-align: justify'>
ntre los tres modelos evaluados, el AE1 (30-20-10-20-30) demuestra la mejor capacidad de discriminación entre fraude y no fraude, alcanzando un AUC de 0.9412, 
ligeramente superior al de AE2 (30-25-15-25-30) con un AUC de 0.93, y muy superior al de AE3 (30-28-26-28-30) con un AUC de 0.82. La diferencia entre AE1 y AE2 es 
poca y ambos modelos llegan a ofrecer un rendimiento aceptable, mientras que la diferencia entre AE1 y AE3 es significativa, indicando que el rendimiento de AE3 es 
considerablemente peor. Esto sugiere que aumentar la capacidad del modelo no necesariamente mejora la detección de fraudes, una arquitectura más grande como la de AE3 
puede llevar al sobreajuste, un indicio de esto es que el 75% de los errores para datos no fraudulentos de AE1 es de 3 unidades y para AE2 es de 6 unidades, esto aumenta
para AE3 llegando a las 12 unidades, lo que nos indica que a comparacion de los otros modelos, este no genarilo bien y se ajusto mucho a la estructura de los datos de 
entrenamiento. En contraste, modelos más compactos como AE1 y AE2 logran generalizar mejor el comportamiento normal y amplifican los errores de reconstrucción frente a 
fraudes, favoreciendo una detección más efectiva. En conclusión, una arquitectura más simple tiende a ofrecer no necesariamente ofrece un mal rendimineto, en este caso 
una arquitectura siemple puede ofrecer mejores resultados.
""", unsafe_allow_html=True) 