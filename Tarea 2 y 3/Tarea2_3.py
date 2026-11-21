import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Cargamos y preparamos los datos

games_df = pd.read_csv("dataset_limpio.csv")

# Eliminar registros duplicados.
games_df_1 = games_df.drop_duplicates().copy()

# Limpiamos espacios en las variables categóricas.
games_df_1["company"] = games_df_1["company"].str.strip()
games_df_1["platform"] = games_df_1["platform"].str.strip()

# Convertimos las puntuaciones a valores numéricos.
games_df_1["user_review"] = pd.to_numeric(games_df_1["user_review"], errors="coerce")

games_df_1["meta_score"] = pd.to_numeric(games_df_1["meta_score"], errors="coerce")

# Obtenemos el año de lanzamiento.
games_df_1["release_year"] = pd.to_datetime(games_df_1["release_date"], errors="coerce").dt.year

# Conservamos los registros que tienen todos los datos
# necesarios para comparar ambos modelos.
columnas_necesarias = ["company", "platform", "release_year", "user_review", "meta_score"]

games_df_1 = games_df_1.dropna(subset=columnas_necesarias).copy()

# Creamos las categorías de valoración.
# Baja: user_review < 7
# Alta: user_review >= 7

games_df_1["review_categoria"] = pd.cut(
    games_df_1["user_review"],
    bins=[-float("inf"), 7, float("inf")],
    labels=["Baja", "Alta"],
    right=False
)

# Mostramos la distribución de categorías.
print("\nDistribución de categorías:")
print(games_df_1["review_categoria"].value_counts().reindex(["Baja", "Alta"]))
print("\nPorcentaje por categoría:")
print((games_df_1["review_categoria"].value_counts(normalize=True).reindex(["Baja", "Alta"]) * 100).round(2))

# Visualizamos la distribución.
games_df_1["review_categoria"].value_counts().reindex(["Baja", "Alta"]).plot(kind="bar", rot=0)

plt.title("Distribución de las valoraciones de usuarios")
plt.xlabel("Categoría")
plt.ylabel("Cantidad de videojuegos")
plt.tight_layout()
plt.show()

# Definimos los modelos

# Modelo 1: sin meta_score.
X_base = games_df_1[["company", "platform", "release_year"]]

# Modelo 2: con meta_score.
X_meta = games_df_1[["company", "platform", "release_year", "meta_score"]]

# Variable objetivo compartida.
y = games_df_1["review_categoria"]

# Dividimos los datos en entrenamiento y prueba

# Realizar una sola división para ambos modelos.
indices_train, indices_test = train_test_split(
    y.index,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Obtener los datos de entrenamiento y prueba.
X_base_train = X_base.loc[indices_train]
X_base_test = X_base.loc[indices_test]

X_meta_train = X_meta.loc[indices_train]
X_meta_test = X_meta.loc[indices_test]

y_train = y.loc[indices_train]
y_test = y.loc[indices_test]

print("\nRegistros de entrenamiento:", len(y_train))
print("Registros de prueba:", len(y_test))

# Valores de K que se evaluarán.
valores_k = [7, 11, 15, 21, 25, 31, 41, 51]

# Funcion que usaremos para entrenar y evaluar KNN

def evaluar_knn(X_train, X_test, nombre_modelo, incluir_meta_score=False):

    tiempo_inicial = time.perf_counter()

    # Definimos las variables que se transformarán.
    variables_categoricas = ["company", "platform"]
    variables_numericas = ["release_year"]

    if incluir_meta_score:
        variables_numericas.append("meta_score")

    # Preparamos las transformaciones.
    preprocesador = ColumnTransformer(
        transformers=[("categoricas", OneHotEncoder(handle_unknown="ignore"), variables_categoricas),
                        ("numericas", StandardScaler(), variables_numericas)])

    # Creamos el pipeline.
    pipeline = Pipeline([
        ("preprocesador", preprocesador),
        ("knn", KNeighborsClassifier())
    ])

    # Buscamos el mejor valor de K.
    busqueda = GridSearchCV(estimator=pipeline, param_grid={"knn__n_neighbors": valores_k}, scoring="f1_macro", cv=5,
        n_jobs=1, return_train_score=True)

    busqueda.fit(X_train, y_train)

    # Obtenemos el mejor modelo y realizamos las predicciones.
    mejor_modelo = busqueda.best_estimator_
    predicciones = mejor_modelo.predict(X_test)

    # Calculamos métricas.
    accuracy = accuracy_score(y_test, predicciones)

    f1_macro = f1_score(y_test, predicciones, labels=["Baja", "Alta"], average="macro", zero_division=0)

    # Mostramos los resultados.
    print("\n" + "=" * 55)
    print(nombre_modelo)
    print("=" * 55)
    print("Mejor K:", busqueda.best_params_["knn__n_neighbors"])
    print("F1 macro en validación:", round(busqueda.best_score_, 4))
    print("Accuracy en prueba:", round(accuracy, 4))
    print("Accuracy en porcentaje:", round(accuracy * 100, 2), "%")
    print("F1 macro en prueba:", round(f1_macro, 4))
    print("\nReporte de clasificación:")
    print(classification_report(y_test, predicciones, labels=["Baja", "Alta"], zero_division=0))

    # Matriz de confusión.
    matriz = confusion_matrix(y_test, predicciones, labels=["Baja", "Alta"])

    visualizador = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=["Baja", "Alta"])

    visualizador.plot(cmap="Blues", values_format="d")
    plt.title(f"Matriz de confusión: {nombre_modelo}")
    plt.tight_layout()
    plt.show()

    # Organizamos los resultados de cada valor de K.
    resultados = pd.DataFrame(busqueda.cv_results_)

    tabla_k = resultados[["param_knn__n_neighbors", "mean_test_score", "std_test_score", "mean_train_score"]
    ].rename(columns={
        "param_knn__n_neighbors": "K",
        "mean_test_score": "F1_macro_validacion",
        "std_test_score": "Desviacion_F1",
        "mean_train_score": "F1_macro_entrenamiento"
    })

    tabla_k["K"] = tabla_k["K"].astype(int)

    tiempo_final = time.perf_counter()

    print("\nTiempo de ejecución:", tiempo_final - tiempo_inicial)

    # Devolvemos los resultados para compararlos.
    return {
        "modelo": mejor_modelo,
        "mejor_k": busqueda.best_params_["knn__n_neighbors"],
        "f1_validacion": busqueda.best_score_,
        "accuracy": accuracy,
        "f1_macro_prueba": f1_macro,
        "tabla_k": tabla_k
    }

# Entrenamos ambos modelos

resultado_base = evaluar_knn(X_base_train, X_base_test, "MODELO 1: Sin meta_score")

resultado_meta = evaluar_knn(X_meta_train, X_meta_test, "MODELO 2: Con meta_score", incluir_meta_score=True)

# Funcionamiento de KNN

# Seleccionamos un videojuego del conjunto de prueba
posicion = 0

# Obtenemos sus características
videojuego = X_meta_test.iloc[[posicion]]

# Obtenemos su categoría real y predicción
categoria_real = y_test.iloc[posicion]
categoria_predicha = resultado_meta["modelo"].predict(videojuego)[0]

# Obtenemos los datos de entrenamiento transformados
modelo = resultado_meta["modelo"]

X_train_transformado = modelo.named_steps["preprocesador"].transform(X_meta_train)
videojuego_transformado = modelo.named_steps["preprocesador"].transform(videojuego)

# Buscamos los K vecinos más cercanos
distancias, indices_vecinos = modelo.named_steps["knn"].kneighbors(videojuego_transformado)

# Recuperamos la información original de los vecinos
vecinos = games_df_1.loc[indices_train].iloc[indices_vecinos[0]].copy()

# Agregamos distancias y categorías
vecinos["distancia"] = distancias[0]
vecinos["categoria"] = y_train.iloc[indices_vecinos[0]].values

# Mostramos la información del videojuego seleccionado
print("VIDEOJUEGO A CLASIFICAR")
print("-----------------------")
print("Nombre:", games_df_1.loc[videojuego.index[0], "name"])
print("Empresa:", videojuego.iloc[0]["company"])
print("Plataforma:", videojuego.iloc[0]["platform"])
print("Año:", videojuego.iloc[0]["release_year"])
print("Meta Score:", videojuego.iloc[0]["meta_score"])
print("Categoría real:", categoria_real)
print("Categoría predicha:", categoria_predicha)

# Mostramos los vecinos más cercanos
print("\nVECINOS MÁS CERCANOS")
print("--------------------")

columnas_mostrar = ["name", "company", "platform", "release_year", "meta_score", "categoria", "distancia"]

print(vecinos[columnas_mostrar].sort_values("distancia").reset_index(drop=True).to_string(index=False))

# Mostramos la votación de los vecinos
print("\nVOTACIÓN DE LOS VECINOS")
print("-----------------------")

votacion = vecinos["categoria"].value_counts()

print(votacion)

print("\nPorcentaje de votos:")
print((votacion / len(vecinos) * 100).round(2))

# Comparación final

comparacion_modelos = pd.DataFrame([
    {
        "Modelo": "Sin meta_score",
        "Mejor K": resultado_base["mejor_k"],
        "F1 macro validación": resultado_base["f1_validacion"],
        "Accuracy prueba": resultado_base["accuracy"],
        "F1 macro prueba": resultado_base["f1_macro_prueba"]
    },
    {
        "Modelo": "Con meta_score",
        "Mejor K": resultado_meta["mejor_k"],
        "F1 macro validación": resultado_meta["f1_validacion"],
        "Accuracy prueba": resultado_meta["accuracy"],
        "F1 macro prueba": resultado_meta["f1_macro_prueba"]
    }
])

print("\n" + "=" * 65)
print("COMPARACIÓN FINAL DE LOS DOS MODELOS")
print("=" * 65)
print(comparacion_modelos.round(4).to_string(index=False))

# Comparamos el F1 macro de cada k evaluada

plt.figure(figsize=(9, 5))
plt.plot(resultado_base["tabla_k"]["K"], resultado_base["tabla_k"]["F1_macro_validacion"], marker="o", label="Sin meta_score")
plt.plot(resultado_meta["tabla_k"]["K"], resultado_meta["tabla_k"]["F1_macro_validacion"], marker="s", label="Con meta_score")
plt.xlabel("Número de vecinos (K)")
plt.ylabel("F1 macro en validación")
plt.title("Comparación de modelos según el valor de K")
plt.xticks(valores_k)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Mostramos los resultados de cada valor de K.
print("\nResultados de K para el modelo sin meta_score:")
print(resultado_base["tabla_k"].round(4).to_string(index=False))

print("\nResultados de K para el modelo con meta_score:")
print(resultado_meta["tabla_k"].round(4).to_string(index=False))

print("\nEl modelo con meta_score obtuvo el mayor F1 macro en prueba.")
