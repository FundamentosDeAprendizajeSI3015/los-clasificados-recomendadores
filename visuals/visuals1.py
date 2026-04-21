import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==============================================================
# CARGA DE DATOS
# Cada dataset viene de una fuente distinta, asi que se cargan
# por separado. Cambia las rutas segun donde tengas los archivos.
# df_usuarios es el dataset combinado que construiste con los
# perfiles de usuario y los targets (_rec) para cada dominio.
# ==============================================================

df_vinos    = pd.read_csv("ruta/vinos.csv")
df_libros   = pd.read_csv("ruta/libros.csv")
df_netflix  = pd.read_csv("ruta/netflix.csv")
df_musica   = pd.read_csv("ruta/musica.csv")
df_usuarios = pd.read_csv("ruta/usuarios.csv")


# ==============================================================
# CONFIGURACION VISUAL
# Se define una paleta de colores y se sobreescriben los defaults
# de matplotlib para que todas las graficas salgan con el mismo
# estilo sin tener que repetir los parametros en cada una.
# ==============================================================

sns.set_theme(style="darkgrid", palette="muted")

PALETTE = ["#c8f060", "#f060a8", "#60d4f0", "#f0a060", "#a060f0", "#60f0b0"]

plt.rcParams.update({
    "figure.facecolor": "#0d0d0f",
    "axes.facecolor":   "#141418",
    "axes.edgecolor":   "#252530",
    "axes.labelcolor":  "#e8e8f0",
    "xtick.color":      "#6a6a80",
    "ytick.color":      "#6a6a80",
    "text.color":       "#e8e8f0",
    "grid.color":       "#1e1e28",
    "grid.linewidth":   0.5,
})


# ==============================================================
# SECCION 1 — EDA VINOS
# Se revisa la distribucion de las variables quimicas principales
# con histogramas. La idea es ver si hay sesgos, valores extremos
# o variables que ya de entrada se ven poco informativas.
# Despues se hace la correlacion para identificar que variables
# se mueven juntas, lo cual es util antes de escalar o hacer PCA.
# ==============================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("EDA — Calidad de Vinos", fontsize=14, y=1.01)

num_cols_vinos = ["fixed acidity", "volatile acidity", "citric acid",
                  "residual sugar", "alcohol", "quality"]

for ax, col in zip(axes.flatten(), num_cols_vinos):
    ax.hist(df_vinos[col].dropna(), bins=30, color=PALETTE[0], alpha=0.75, edgecolor="#0d0d0f")
    ax.set_title(col, fontsize=9)

plt.tight_layout()
plt.savefig("eda_vinos.png", dpi=150, bbox_inches="tight")
plt.show()


# Heatmap de correlacion entre todas las variables numericas del dataset.
# Valores cercanos a 1 o -1 indican relacion lineal fuerte entre columnas.
# Sirve para detectar multicolinealidad antes de meter todo a un modelo.

fig, ax = plt.subplots(figsize=(10, 8))
cols_corr = [c for c in df_vinos.columns if df_vinos[c].dtype in [float, int]]
corr = df_vinos[cols_corr].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGn", ax=ax,
            linewidths=0.4, linecolor="#0d0d0f", annot_kws={"size": 8})
ax.set_title("Correlacion — Vinos", fontsize=12)
plt.tight_layout()
plt.savefig("corr_vinos.png", dpi=150, bbox_inches="tight")
plt.show()


# ==============================================================
# SECCION 2 — EDA MUSICA
# Las audio features de Spotify van en escala 0-1 (excepto
# loudness y tempo), entonces los histogramas sirven para ver
# si la distribucion es uniforme, concentrada o bimodal.
# El boxplot de valence por continente muestra si el "mood"
# de la musica varia geograficamente, lo cual es relevante
# porque el dataset tiene region y continent como variables.
# ==============================================================

audio_features = ["danceability", "energy", "loudness", "speechiness",
                  "acousticness", "instrumentalness", "liveness", "valence", "tempo"]

# Se filtran solo las columnas que realmente existen en el archivo,
# por si alguna viene con nombre distinto o no esta incluida.
audio_features = [c for c in audio_features if c in df_musica.columns]

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle("EDA — Features de Audio", fontsize=14)

for ax, col in zip(axes.flatten(), audio_features):
    ax.hist(df_musica[col].dropna(), bins=30, color=PALETTE[2], alpha=0.75, edgecolor="#0d0d0f")
    ax.set_title(col, fontsize=9)

# Si sobran subplots (porque hay menos de 9 features), se ocultan.
for ax in axes.flatten()[len(audio_features):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig("eda_musica.png", dpi=150, bbox_inches="tight")
plt.show()


# Boxplot de valence agrupado por continente, ordenado por mediana.
# Solo se ejecuta si el dataset tiene esas dos columnas.
if "continent" in df_musica.columns and "valence" in df_musica.columns:
    fig, ax = plt.subplots(figsize=(12, 5))
    orden = df_musica.groupby("continent")["valence"].median().sort_values().index
    sns.boxplot(data=df_musica, x="continent", y="valence", order=orden,
                palette=PALETTE, ax=ax)
    ax.set_title("Valence por Continente")
    plt.tight_layout()
    plt.savefig("valence_continente.png", dpi=150, bbox_inches="tight")
    plt.show()


# ==============================================================
# SECCION 3 — EDA LIBROS
# Este dataset ya viene con features construidas (scores, tasas,
# promedios por usuario), no son variables crudas. Los histogramas
# muestran como se distribuyen esos indicadores entre usuarios.
# El scatter de engagement vs abandono es clave: en teoria deberian
# tener correlacion negativa, y si no la tienen hay algo raro en
# como se construyeron esas variables.
# ==============================================================

libro_cols = ["engagement_score", "abandono_score", "duracion_promedio",
              "velocidad_lectura", "tasa_abandono", "progreso_promedio"]

# Igual que en musica, se filtra por si alguna columna no existe.
libro_cols = [c for c in libro_cols if c in df_libros.columns]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("EDA — Sesiones de Lectura", fontsize=14)

for ax, col in zip(axes.flatten(), libro_cols):
    ax.hist(df_libros[col].dropna(), bins=30, color=PALETTE[1], alpha=0.75, edgecolor="#0d0d0f")
    ax.set_title(col, fontsize=9)

for ax in axes.flatten()[len(libro_cols):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig("eda_libros.png", dpi=150, bbox_inches="tight")
plt.show()


# Scatter engagement vs abandono. Cada punto es un usuario.
# Si el modelo esta bien construido, usuarios con alto engagement
# deberian tener bajo abandono y viceversa.
if "abandono_score" in df_libros.columns and "engagement_score" in df_libros.columns:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df_libros["engagement_score"], df_libros["abandono_score"],
               alpha=0.4, color=PALETTE[1], edgecolors="none", s=20)
    ax.set_xlabel("engagement_score")
    ax.set_ylabel("abandono_score")
    ax.set_title("Engagement vs Abandono — Libros")
    plt.tight_layout()
    plt.savefig("scatter_libros.png", dpi=150, bbox_inches="tight")
    plt.show()


# ==============================================================
# SECCION 4 — EDA NETFLIX
# Se revisa la distribucion de scores de IMDB y el conteo por
# tipo de contenido (pelicula vs serie). Son las dos variables
# mas directas para entender de que esta compuesto el catalogo.
# ==============================================================

if "imdb_score" in df_netflix.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("EDA — Netflix / IMDB", fontsize=14)

    axes[0].hist(df_netflix["imdb_score"].dropna(), bins=30,
                 color=PALETTE[3], alpha=0.75, edgecolor="#0d0d0f")
    axes[0].set_title("Distribucion IMDB Score")

    if "type_base" in df_netflix.columns:
        conteo = df_netflix["type_base"].value_counts()
        axes[1].bar(conteo.index, conteo.values, color=PALETTE[3], alpha=0.8)
        axes[1].set_title("Tipo de contenido")

    plt.tight_layout()
    plt.savefig("eda_netflix.png", dpi=150, bbox_inches="tight")
    plt.show()


# ==============================================================
# SECCION 5 — CLUSTERING SOBRE EL DATASET DE USUARIOS
# Aqui es donde se trabaja el dataset combinado. La idea es
# encontrar grupos naturales de usuarios segun su perfil, sin
# usar los targets. Despues se revisa si cada cluster tiene
# preferencias distintas en terminos de generos recomendados.
# ==============================================================

# Se toman solo las columnas numericas y se excluyen los targets
# (_rec) porque esos son la salida del modelo, no la entrada.
feat_cols   = df_usuarios.select_dtypes(include=[np.number]).columns.tolist()
target_cols = [c for c in df_usuarios.columns if "_rec" in c]
feat_cols   = [c for c in feat_cols if c not in target_cols]

# Se eliminan filas con nulos para que KMeans no falle,
# y se escala con StandardScaler porque KMeans es sensible
# a la magnitud de las variables.
X        = df_usuarios[feat_cols].dropna()
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)


# --- Elbow + Silhouette para elegir K ---
# Se prueba KMeans con K de 2 a 8. La inercia (elbow) baja
# siempre con mas clusters, el punto de quiebre sugiere el K optimo.
# El silhouette mide que tan bien separados estan los clusters,
# el valor mas alto es el mejor.

k_range     = range(2, 9)
inertias    = []
silhouettes = []

for k in k_range:
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Seleccion de K — Dataset Usuarios", fontsize=13)

axes[0].plot(list(k_range), inertias, marker="o", color=PALETTE[0], linewidth=2)
axes[0].set_title("Elbow Method (Inercia)")
axes[0].set_xlabel("K")
axes[0].set_ylabel("Inercia")

axes[1].plot(list(k_range), silhouettes, marker="o", color=PALETTE[2], linewidth=2)
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Score")

plt.tight_layout()
plt.savefig("elbow_silhouette.png", dpi=150, bbox_inches="tight")
plt.show()


# --- KMeans final ---
# Cambia K_OPTIMO segun lo que veas en el elbow y el silhouette.
# No hay una respuesta exacta, es una decision interpretativa.

K_OPTIMO = 3

km_final       = KMeans(n_clusters=K_OPTIMO, random_state=42, n_init=10)
cluster_labels = km_final.fit_predict(X_scaled)

# Se agrega la columna de cluster al dataframe original
# usando el indice de X (que puede ser subset si habia nulos).
df_usuarios_cl            = df_usuarios.loc[X.index].copy()
df_usuarios_cl["cluster"] = cluster_labels


# --- PCA 2D ---
# PCA reduce las dimensiones a 2 componentes principales para
# poder visualizar los clusters en un plano. Los ejes muestran
# que porcentaje de la varianza total explica cada componente.
# Si entre PC1 y PC2 suman poco (menos del 50%), la separacion
# visual puede ser enganosa.

pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(9, 7))
for k in range(K_OPTIMO):
    mask = cluster_labels == k
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               label=f"Cluster {k}", alpha=0.6, s=30,
               color=PALETTE[k % len(PALETTE)], edgecolors="none")

ax.set_title(f"PCA 2D — {K_OPTIMO} Clusters")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.legend()
plt.tight_layout()
plt.savefig("pca_clusters.png", dpi=150, bbox_inches="tight")
plt.show()


# --- t-SNE 2D ---
# t-SNE es una alternativa a PCA para visualizacion. No conserva
# distancias globales pero si agrupa puntos similares localmente,
# por eso a veces muestra separaciones mas claras que PCA.
# La perplexity se limita al minimo entre 30 y n-1 para evitar
# errores con datasets pequenos.

tsne   = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled) - 1))
X_tsne = tsne.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(9, 7))
for k in range(K_OPTIMO):
    mask = cluster_labels == k
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               label=f"Cluster {k}", alpha=0.6, s=30,
               color=PALETTE[k % len(PALETTE)], edgecolors="none")

ax.set_title(f"t-SNE 2D — {K_OPTIMO} Clusters")
ax.legend()
plt.tight_layout()
plt.savefig("tsne_clusters.png", dpi=150, bbox_inches="tight")
plt.show()


# --- Varianza explicada acumulada ---
# Se hace PCA completo (sin limitar componentes) para ver cuantos
# componentes hacen falta para explicar el 90% o 95% de la varianza.
# Esto ayuda a decidir si vale la pena reducir dimensionalidad
# antes de entrenar el modelo.

pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)
var_acum = np.cumsum(pca_full.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1, len(var_acum) + 1), var_acum, marker="o", color=PALETTE[0], linewidth=2)
ax.axhline(0.90, color=PALETTE[1], linestyle="--", linewidth=1, label="90% varianza")
ax.axhline(0.95, color=PALETTE[3], linestyle="--", linewidth=1, label="95% varianza")
ax.set_title("Varianza Explicada Acumulada — PCA")
ax.set_xlabel("Numero de componentes")
ax.set_ylabel("Varianza acumulada")
ax.legend()
plt.tight_layout()
plt.savefig("pca_varianza.png", dpi=150, bbox_inches="tight")
plt.show()


# --- Heatmap de perfil por cluster ---
# Se calcula el promedio de cada variable numerica por cluster
# y se normaliza entre 0 y 1 para poder comparar variables que
# estan en escalas distintas. Sirve para describir cada cluster:
# cual tiene usuarios con mas engagement, mas velocidad de lectura, etc.

perfil      = df_usuarios_cl.groupby("cluster")[feat_cols].mean()
perfil_norm = (perfil - perfil.min()) / (perfil.max() - perfil.min() + 1e-9)

fig, ax = plt.subplots(figsize=(max(12, len(feat_cols) * 0.6), 4))
sns.heatmap(perfil_norm, annot=True, fmt=".2f", cmap="YlGn",
            linewidths=0.3, linecolor="#0d0d0f", ax=ax, annot_kws={"size": 7})
ax.set_title("Perfil Normalizado por Cluster", fontsize=12)
plt.tight_layout()
plt.savefig("perfil_clusters.png", dpi=150, bbox_inches="tight")
plt.show()


# --- Distribucion de targets por cluster ---
# Para cada columna _rec (genero libro, tipo vino, genero musical,
# genero serie) se muestra que proporcion de cada cluster cae en
# cada categoria. Si los clusters son buenos, deberian tener
# distribuciones distintas entre si.

fig, axes = plt.subplots(1, len(target_cols), figsize=(5 * len(target_cols), 5))

# Si solo hay un target, axes no es lista, se convierte para que
# el loop de abajo funcione igual en cualquier caso.
if len(target_cols) == 1:
    axes = [axes]

for ax, tc in zip(axes, target_cols):
    ct      = df_usuarios_cl.groupby(["cluster", tc]).size().unstack(fill_value=0)
    ct_norm = ct.div(ct.sum(axis=1), axis=0)  # proporciones en vez de conteos absolutos
    ct_norm.plot(kind="bar", ax=ax, color=PALETTE[:len(ct_norm.columns)], alpha=0.85)
    ax.set_title(tc.replace("_rec", "").replace("_", " ").title())
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proporcion")
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=0)

plt.suptitle("Distribucion de Recomendaciones por Cluster", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("targets_por_cluster.png", dpi=150, bbox_inches="tight")
plt.show()

print("Todas las graficas generadas y guardadas.")
print("Codigo generado por Claudia y comentado por mi.")