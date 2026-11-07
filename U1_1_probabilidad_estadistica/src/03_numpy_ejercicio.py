import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# ============================================================
# 1. CARGAR LOS DATOS
# ============================================================

# Fetch the data.
df = pd.read_csv("https://ourworldindata.org/grapher/children-born-per-woman.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

# Fetch the metadata
metadata = requests.get("https://ourworldindata.org/grapher/children-born-per-woman.metadata.json?v=1&csvType=full&useColumnShortNames=true").json()

print("Primeras filas:")
print(df.head())

print("\nDimensiones:")
print(df.shape)

print("\nColumnas:")
print(df.columns.tolist())

print("\nTipos de datos:")
print(df.dtypes)

print("\nEstadísticas:")
print(df.describe())


# ============================================================
# 2. EXPLORAR LAS ENTIDADES
# ============================================================

print("\nEntidades disponibles:")
print(df["Entity"].unique())


# ============================================================
# 3. FILTRAR MÉXICO
# ============================================================

mexico = df[df["Entity"] == "Mexico"].copy()

print("\nDatos de México:")
print(mexico.head())

print("\nÚltimos registros de México:")
print(mexico.tail())

print("\nDimensiones México:")
print(mexico.shape)


# ============================================================
# 4. FILTRAR JAPÓN
# ============================================================

japon = df[df["Entity"] == "Japan"].copy()

print("\nDatos de Japón:")
print(japon.head())

print("\nÚltimos registros de Japón:")
print(japon.tail())

print("\nDimensiones Japón:")
print(japon.shape)


# ============================================================
# 5. IDENTIFICAR LA COLUMNA DE NATALIDAD
# ============================================================

print("\nColumnas:")
print(df.columns.tolist())


# ============================================================
# 6. CONVERTIR LOS DATOS A NUMPY
# ============================================================

# La columna del indicador se llama:
# "Birth rate - UN WPP"

anios_mexico = mexico["Year"].to_numpy()

tasa_mexico = mexico["Birth rate - UN WPP"].to_numpy()

anios_japon = japon["Year"].to_numpy()

tasa_japon = japon["Birth rate - UN WPP"].to_numpy()


print("\nAños México:")
print(anios_mexico)

print("\nTasas México:")
print(tasa_mexico)


# ============================================================
# 7. ESTADÍSTICA DE MÉXICO
# ============================================================

promedio_mexico = np.mean(tasa_mexico)
maximo_mexico = np.max(tasa_mexico)
minimo_mexico = np.min(tasa_mexico)
desviacion_mexico = np.std(tasa_mexico)

print("\n===== MÉXICO =====")

print("Promedio:", promedio_mexico)
print("Máximo:", maximo_mexico)
print("Mínimo:", minimo_mexico)
print("Desviación estándar:", desviacion_mexico)


# ============================================================
# 8. ESTADÍSTICA DE JAPÓN
# ============================================================

promedio_japon = np.mean(tasa_japon)
maximo_japon = np.max(tasa_japon)
minimo_japon = np.min(tasa_japon)
desviacion_japon = np.std(tasa_japon)

print("\n===== JAPÓN =====")

print("Promedio:", promedio_japon)
print("Máximo:", maximo_japon)
print("Mínimo:", minimo_japon)
print("Desviación estándar:", desviacion_japon)


# ============================================================
# 9. COMPARAR PROMEDIOS
# ============================================================

diferencia_promedio = promedio_mexico - promedio_japon

print("\n===== COMPARACIÓN =====")

print("Promedio México:", promedio_mexico)
print("Promedio Japón:", promedio_japon)
print("Diferencia:", diferencia_promedio)


# ============================================================
# 10. GRÁFICA DE MÉXICO
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    anios_mexico,
    tasa_mexico,
    label="México"
)

plt.title("Evolución de la tasa de natalidad de México")
plt.xlabel("Año")
plt.ylabel("Nacimientos por cada 1,000 habitantes")
plt.grid(True)
plt.legend()

plt.show()


# ============================================================
# 11. COMPARACIÓN MÉXICO VS JAPÓN
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    anios_mexico,
    tasa_mexico,
    label="México"
)

plt.plot(
    anios_japon,
    tasa_japon,
    label="Japón"
)

plt.title("Comparación de la tasa de natalidad")
plt.xlabel("Año")
plt.ylabel("Nacimientos por cada 1,000 habitantes")
plt.grid(True)
plt.legend()

plt.show()


# ============================================================
# 12. CALCULAR DIFERENCIA ENTRE MÉXICO Y JAPÓN
# ============================================================

# Como ambos arreglos tienen los mismos años,
# podemos realizar una operación elemento por elemento.

diferencia = tasa_mexico - tasa_japon

print("\nDiferencia México - Japón:")
print(diferencia)


# ============================================================
# 13. GRÁFICA DE LA DIFERENCIA
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    anios_mexico,
    diferencia,
    label="México - Japón"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.title("Diferencia en la tasa de natalidad: México vs. Japón")
plt.xlabel("Año")
plt.ylabel("Diferencia en nacimientos por cada 1,000 habitantes")
plt.grid(True)
plt.legend()

plt.show()


# ============================================================
# 14. IDENTIFICAR EL AÑO DE MAYOR Y MENOR DIFERENCIA
# ============================================================

indice_max = np.argmax(diferencia)
indice_min = np.argmin(diferencia)

print("\n===== DIFERENCIAS =====")

print(
    "Mayor diferencia:",
    diferencia[indice_max]
)

print(
    "Año de mayor diferencia:",
    anios_mexico[indice_max]
)

print(
    "Menor diferencia:",
    diferencia[indice_min]
)

print(
    "Año de menor diferencia:",
    anios_mexico[indice_min]
)