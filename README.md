# Trabajo Práctico Final — Aprendizaje de Máquina

**Clasificación multiclase de la severidad legal de denuncias del NYPD**

**Materia:** Aprendizaje de Máquina — CEIA 27Co2026  
**Integrantes:** Maria Celeste Corominas — Christian Marcelo Yanez Flores  
**Grupo:** 4  
**Fecha:** 17/10/2026  
**Notebook:** [Abrir en Google Colab](https://colab.research.google.com/drive/1ylfDfGQyPETLeykbIeO3paadqm9Z620U?usp=sharing)

Este trabajo estudia cuánto puede inferirse sobre la categoría legal registrada de una denuncia del NYPD utilizando únicamente información contextual, sin incorporar variables que describen el delito de manera tan directa que produzcan *data leakage*.

La pregunta central es:

> **¿Hasta qué punto puede inferirse `LAW_CAT_CD` a partir de variables contextuales, excluyendo las variables que codifican directa o casi directamente el delito?**

**Dataset**

Se utiliza **NYPD Complaint Data Current (Year To Date)**, publicado por NYC Open Data y provisto por el New York City Police Department (NYPD), con identificador `5uac-w243`.

El período de estudio comprende denuncias reportadas entre el **01/01/2026 y el 30/06/2026**. El snapshot utilizado contiene **279.513 registros y 36 variables**.

Fuente oficial:

- https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Current-Year-To-Date-/5uac-w243
- https://data.cityofnewyork.us/resource/5uac-w243.csv

La variable objetivo es `LAW_CAT_CD`, con tres categorías:

- **VIOLATION:** infracción para la cual no puede imponerse una pena de prisión superior a 15 días.
- **MISDEMEANOR:** delito para el cual puede imponerse una pena superior a 15 días, pero no superior a un año.
- **FELONY:** delito para el cual puede imponerse una pena superior a un año.

En términos simplificados, las categorías representan niveles crecientes de severidad penal:

`VIOLATION → MISDEMEANOR → FELONY`

La distribución observada es aproximadamente:

| Clase | Proporción |
|---|---:|
| VIOLATION | 16,36 % |
| MISDEMEANOR | 52,15 % |
| FELONY | 31,49 % |

Como la fuente *Current YTD* es dinámica, la notebook prioriza el uso de un snapshot congelado:

```text
datasets/NYPD_Complaint_Data_Current_2026_Q2.csv
```

La reproducibilidad no depende únicamente del código y de `random_state`: también requiere conservar la versión exacta de los datos utilizada en el experimento.

**Planteo metodológico**

El dataset contiene variables muy próximas a la propia definición del delito. El análisis empírico de la notebook muestra que:

- `PD_CD`: 357/357 categorías aparecen asociadas a una única clase de `LAW_CAT_CD`;
- `KY_CD`: 63/66 categorías aparecen asociadas a una única clase;
- `OFNS_DESC`: 42/55 categorías aparecen asociadas a una única clase.

Por este motivo, se excluyen del modelado:

```text
PD_CD
PD_DESC
KY_CD
OFNS_DESC
```

También se excluye `CMPLNT_NUM`, porque es un identificador y no una característica explicativa.

El objetivo no es obtener métricas artificialmente altas, sino medir cuánta señal predictiva permanece cuando se retira la información que prácticamente contiene la respuesta.

**Predictores utilizados**

El modelo principal trabaja con 15 variables contextuales:

```text
LATITUDE
LONGITUDE
DEMORA_REPORTE_DIAS
CMPLNT_FR_MONTH
CMPLNT_FR_DOW
CMPLNT_FR_HOUR
CRM_ATPT_CPTD_CD
LOC_OF_OCCUR_DESC
PREM_TYP_DESC
HADEVELOPT
STATION_NAME
PARKS_NM
HOUSING_PSA
TRANSIT_DISTRICT
JURIS_DESC
```

Se excluyen además atributos sensibles de víctima y sospechoso.

Después de la limpieza determinística se conservaron **279.495 observaciones** para modelado. La partición estratificada quedó formada por:

- **Train:** 223.596 observaciones.
- **Test:** 55.899 observaciones.

**Preprocesamiento**

Las transformaciones aprendidas se implementan dentro de un `Pipeline` de *scikit-learn* para evitar que información de validación intervenga en el ajuste.

La estrategia incluye:

- escalado de coordenadas;
- transformación `log1p` y estandarización de la demora de reporte;
- representación cíclica de mes, hora y día de la semana;
- codificación binaria de la condición consumado/tentado;
- *Frequency Encoding* para variables categóricas de alta cardinalidad;
- *One-Hot Encoding* para `LOC_OF_OCCUR_DESC`;
- conservación de las 20 categorías más frecuentes de `PREM_TYP_DESC`, agrupando las restantes como `OTRO`, seguida de *One-Hot Encoding*.

El preprocesamiento produce **43 features finales**.

**Baselines**

Se utilizan dos referencias:

- `DummyClassifier(strategy="most_frequent")`, que siempre predice la clase mayoritaria;
- un baseline contextual que, para cada `PREM_TYP_DESC`, predice la clase modal observada en entrenamiento.

El segundo baseline permite exigir algo más que simplemente superar la clase mayoritaria.

**Modelos comparados**

Se evaluaron siete alternativas bajo el mismo esquema de validación:

- baseline de clase mayoritaria;
- baseline contextual por tipo de premisa;
- regresión logística multinomial;
- árbol de decisión;
- bagging;
- random forest;
- gradient boosting.

KNN, SVM con kernel, clustering, PCA y XGBoost fueron considerados, pero no forman parte de la comparación principal por razones metodológicas, computacionales o de interpretabilidad documentadas en la notebook.

**Métrica principal y validación**

La métrica principal es **F1 macro**, porque las tres categorías tienen importancia conceptual y la distribución de clases no es uniforme.

Como métricas complementarias se utilizan:

- balanced accuracy;
- accuracy;
- precision y recall por clase;
- ROC-AUC multiclase One-vs-Rest;
- log-loss.

La selección se realiza sobre una muestra estratificada de **50.000 observaciones de train**, mediante validación cruzada estratificada de **3 folds**. El conjunto de test permanece cerrado hasta finalizar la selección.

Resultados principales de validación cruzada:

| Modelo | F1 macro CV | Accuracy CV |
|---|---:|---:|
| Random Forest | **0,4592** | 0,5040 |
| Gradient Boosting | 0,4522 | 0,4587 |
| Regresión logística | 0,4389 | 0,4459 |
| Bagging | 0,4318 | 0,5122 |
| Árbol | 0,4232 | 0,4300 |
| Baseline contextual | 0,2862 | 0,5254 |
| Baseline mayoritaria | 0,2285 | 0,5215 |

**Balanceo de clases**

No se aplican *undersampling*, *oversampling* ni SMOTE. Se mantiene la distribución real y se estudia explícitamente el efecto de `class_weight`.

En el experimento controlado con Gradient Boosting, al pasar de una configuración sin balanceo a una balanceada:

- accuracy cambió en **−0,0821**;
- balanced accuracy cambió en **+0,1244**;
- F1 macro cambió en **+0,1005**.

Esto muestra el trade-off entre maximizar el porcentaje global de aciertos y repartir mejor el desempeño entre las tres clases.

**Ajuste de hiperparámetros y selección final**

La mejor familia antes del tuning fue **Random Forest**.

Se realizó una búsqueda aleatoria acotada de hiperparámetros. La mejora obtenida fue de aproximadamente **dos milésimas de F1 macro**, por lo que se comparó esa ganancia con la variabilidad entre folds.

Luego se aplicó una regla de **un error estándar (1-SE)**. Tanto el Random Forest base como la versión ajustada quedaron dentro del umbral y se seleccionó el modelo base por parsimonia.

**Modelo final: `Random_Forest`**

**Resultados sobre test**

El conjunto de test se utiliza recién después de cerrar la selección.

| Modelo | Accuracy | Balanced accuracy | F1 macro |
|---|---:|---:|---:|
| Baseline mayoritaria | 0,5215 | 0,3333 | 0,2285 |
| Baseline contextual | 0,5258 | 0,3444 | 0,2642 |
| **Random Forest** | **0,5107** | **0,4819** | **0,4741** |

Métricas adicionales del modelo final:

| Métrica | Valor |
|---|---:|
| Precision macro | 0,4703 |
| Recall macro | 0,4819 |
| F1 weighted | 0,5156 |
| ROC-AUC macro OvR | 0,6812 |
| Log loss | 0,9437 |

El modelo final no maximiza `accuracy`. Su aporte aparece en F1 macro y balanced accuracy, es decir, en una distribución más equilibrada del desempeño entre clases.

Respecto de los baselines:

- Δ F1 macro vs. clase mayoritaria: **+0,2456**;
- Δ F1 macro vs. baseline contextual: **+0,2100**;
- Δ accuracy vs. clase mayoritaria: **−0,0108**;
- Δ accuracy vs. baseline contextual: **−0,0151**.

**Resultados por clase**

| Clase | Precision | Recall | F1 |
|---|---:|---:|---:|
| VIOLATION | 0,3324 | 0,4161 | 0,3696 |
| MISDEMEANOR | 0,6257 | 0,5733 | 0,5984 |
| FELONY | 0,4527 | 0,4562 | 0,4545 |

`VIOLATION` es la clase con menor desempeño. La confusión absoluta más frecuente es:

**MISDEMEANOR → FELONY: 7.589 casos**

**Discriminación probabilística**

El ROC-AUC One-vs-Rest por clase fue:

| Clase | AUC |
|---|---:|
| VIOLATION | 0,7311 |
| FELONY | 0,6613 |
| MISDEMEANOR | 0,6512 |

El ROC-AUC macro del modelo final fue **0,6812**.

Estas métricas evalúan capacidad de discriminación, no calibración de probabilidades.

**Incertidumbre**

Se aplicó bootstrap sobre el conjunto de test con **1.000 remuestreos**.

- F1 macro: **0,4741**
- IC bootstrap 95 %: **[0,4699; 0,4783]**

El intervalo permite acompañar la estimación puntual con una medida explícita de incertidumbre.

**Interpretabilidad**

La importancia por permutación se calcula sobre el `Pipeline` completo, utilizando una muestra reproducible de 3.000 observaciones de test y tres repeticiones por variable.

Las variables contextuales con mayor caída media de F1 macro al ser permutadas fueron:

| Variable | Importancia media |
|---|---:|
| PREM_TYP_DESC | 0,07670 |
| LOC_OF_OCCUR_DESC | 0,03552 |
| LATITUDE | 0,01476 |
| DEMORA_REPORTE_DIAS | 0,01461 |
| CMPLNT_FR_HOUR | 0,00999 |
| LONGITUDE | 0,00916 |

La importancia predictiva no implica causalidad.

**Conclusión**

Los resultados respaldan las tres hipótesis planteadas en la notebook:

- existe señal contextual por encima de los baselines;
- los modelos capaces de representar relaciones no lineales aportan información frente a la referencia lineal;
- una mayor complejidad o un tuning adicional no garantiza una mejora sustantiva.

El resultado principal no es que pueda predecirse la severidad legal con alta precisión. La conclusión es más acotada: **una vez eliminadas las variables que contienen información directa sobre el delito, todavía permanece señal predictiva en el contexto, pero su capacidad discriminativa es limitada**.

El trabajo muestra también que una evaluación defendible requiere controlar *data leakage*, comparar contra baselines, mantener test cerrado durante la selección, estudiar el desbalance de clases y reportar tanto las mejoras como sus costos.

**Limitaciones y próximos pasos**

Entre las limitaciones se consideran:

- `LAW_CAT_CD` es un registro legal y no una variable causal;
- los datos policiales reflejan procesos de denuncia, registro y actuación institucional;
- pueden persistir proxies indirectos aun después de excluir variables sensibles y de leakage;
- la validación aleatoria no reproduce completamente un escenario temporal de producción;
- el recurso *Current YTD* es mutable.

Como extensiones se proponen:

- validación temporal;
- *ablation studies* por bloques de variables;
- comparación con XGBoost;
- búsqueda de hiperparámetros con Optuna;
- algoritmos con soporte categórico nativo;
- calibración de probabilidades;
- análisis de *drift*;
- evaluación de *fairness*.

**Reproducibilidad**

Entorno registrado en la ejecución actual:

```text
Python 3.13.16
pandas 2.2.3
NumPy 2.1.3
scikit-learn 1.6.1
```

Semilla global:

```python
RANDOM_STATE = 42
```

Para reproducir el trabajo se recomienda conservar juntos:

```text
TP_Final_Amq_COROMINAS-YANEZ FLORES.ipynb
README.md
datasets/
└── NYPD_Complaint_Data_Current_2026_Q2.csv
```

**Referencias**

- NYC Open Data / NYPD. *NYPD Complaint Data Current (Year To Date)*.  
  https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Current-Year-To-Date-/5uac-w243

- New York State Senate. *New York Penal Law § 10.00*.  
  https://www.nysenate.gov/legislation/laws/PEN/10.00

- FIUBA — Aprendizaje de Máquina, CEIA.  
  https://github.com/FIUBA-Posgrado-Inteligencia-Artificial/aprMaqI_CEIA

- Bruce, P., Bruce, A. y Gedeck, P. (2020). *Practical Statistics for Data Scientists: 50+ Essential Concepts Using R and Python*. 2nd ed. O'Reilly Media.

- James, G., Witten, D., Hastie, T., Tibshirani, R. y Taylor, J. (2023). *An Introduction to Statistical Learning: with Applications in Python*. Springer.

- Hastie, T., Tibshirani, R. y Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2nd ed. Springer.

- Breiman, L. (2001). “Random Forests”. *Machine Learning*, 45, 5–32.  
  https://doi.org/10.1023/A:1010933404324

- scikit-learn developers. *User Guide and API Reference*.  
  https://scikit-learn.org/stable/
