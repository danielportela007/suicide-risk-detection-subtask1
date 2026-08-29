# Sistema `PostFrasesV1` para la Tarea 1

**Alcance:** Subtask 1: Suicide Risk Detection  
**Sistema:** `PostFrasesV1`  
**Fecha de la ejecución registrada:** 26 de agosto de 2026  
**Estado:** validación OOF exploratoria; no es una evaluación del test oculto

## 1. Objetivo y alcance

El sistema recibe un post de Reddit y produce dos resultados: un nivel de riesgo y las frases del post que justifican la predicción. Las cuatro clases son:

| Clase | Definición operativa |
|---|---|
| `Indicator` | El post no contiene menciones explícitas de suicidio. |
| `Ideation` | Contiene expresiones suicidas explícitas, pero no un plan. |
| `Behavior` | Contiene expresiones suicidas explícitas y señales de autolesión o de un plan. |
| `Attempt` | Menciona explícitamente un intento suicida reciente o pasado. |

La implementación se limita a la Tarea 1 descrita en [bigdatacompetition.cn](https://www.bigdatacompetition.cn/). No entrena ni evalúa factores de riesgo o protección. La columna `factors` se conserva porque pertenece al esquema del CSV, pero contiene `[]` en las 378 filas.

## 2. Resultado principal

| Métrica OOF | Resultado |
|---|---:|
| Risk Weighted F1 | **0.625747** |
| Risk Macro F1 | 0.551130 |
| Accuracy | 0.621407 |
| Evidence Phrase F1 | **0.493796** |
| Contribución ponderada de la Tarea 1 | 0.398438 de 0.70 |
| Score de Tarea 1 normalizado | **0.569196** |

El score se calculó así:

```text
0.40 × 0.6257467580 + 0.30 × 0.4937961088 = 0.3984375358
0.3984375358 / 0.70 = 0.5691964798
```

Las cifras proceden de validación cruzada sobre train y también se usaron para seleccionar configuraciones. No son resultados de leaderboard ni de un holdout intacto.

## 3. Flujo del sistema

```text
Workbooks aumentados
  ├─ validación de esquema, usuarios, cronología y frases
  ├─ post original → MPNet → embedding global de 768 dimensiones
  └─ post_frases → MPNet → similitud con anclas macro e individuales
                              ↓
                    cinco bloques, 896 variables
                              ↓
 StandardScaler → SelectKBest(ANOVA, k=128) → regresión logística
                              ↓
                    nivel de riesgo predicho
                              ↓
 candidatos con offsets del post → similitud con anclas de la clase
                              ↓
          umbral 0.40 → máximo 12 tokens → top 2 → evidencia
```

El riesgo se modela mediante aprendizaje supervisado sobre características semánticas. La evidencia se recupera por similitud con anclas, condicionada por la clase predicha.

## 4. Datos

### 4.1 Archivos y separación

| Partición | Archivo | Filas | Usuarios | SHA-256 |
|---|---|---:|---:|---|
| Train | `data/raw/train dataset-con-frases-con-segmentosV1.xlsx` | 1,635 | 153 | `10d3cd4384336bfbea73c6ad8cd04f53018e12e961038207b22d7703d7e583b5` |
| Test | `data/raw/test dataset-con-frases.xlsx` | 378 | 36 | `2c2746e44159a49fa73e21cb972abc942507e6d5b26417278094584e6da4414d` |

Train y test no comparten usuarios. Los `post_id` son consecutivos desde cero dentro de cada usuario y no se detectaron problemas de cronología. En la carpeta aislada, ambos workbooks son enlaces simbólicos a los archivos canónicos; el texto restringido no se duplicó ni modificó.

### 4.2 Columnas consumidas

Train requiere `row_id`, `anon_user_id`, `post_id`, `post`, `suicide risk`, `evidence for suicide risk level` y `post_frases`. Test requiere las columnas de identificación/texto y `post_frases`. Las columnas de segmentos no intervienen.

### 4.3 Distribución de riesgo

| Clase | Posts | Porcentaje |
|---|---:|---:|
| Indicator | 611 | 37.37 % |
| Ideation | 519 | 31.74 % |
| Behavior | 391 | 23.91 % |
| Attempt | 114 | 6.97 % |
| **Total** | **1,635** | **100 %** |

El desbalance de `Attempt` motivó el uso de pesos de clase balanceados.

### 4.4 Frases suministradas

`post_frases` contiene listas serializadas de frases:

- train: 9,276 frases;
- test: 2,395 frases;
- alineación literal sin distinguir mayúsculas/minúsculas: 100 %.

Las listas se parsean con `ast.literal_eval`. El pipeline falla si falta la columna, si una fila queda sin unidades o si una frase no aparece en su post. Estas frases se usan para crear características de riesgo, pero no se copian directamente como evidencia.

### 4.5 Evidencia gold

Las anotaciones se separan por punto y coma. Los marcadores vacíos, `none`, `nan`, `null` y `n/a` se interpretan como ausencia de evidencia.

| Diagnóstico | Cantidad |
|---|---:|
| Posts con gold no vacío | 1,047 |
| Posts con gold vacío | 588 |
| Spans gold | 1,833 |
| Spans literalmente alineables sin distinguir mayúsculas/minúsculas | 1,660 (90.56 %) |
| Celdas gold ausentes | 5 |

Los 173 spans no alineables impiden asumir offsets exactos para todas las anotaciones. El sistema no corrige ni sobrescribe el post para compensar esas diferencias.

## 5. Preparación y control de calidad

`scripts/01_validate_data.py` comprueba columnas, unicidad de `row_id`, campos obligatorios, etiquetas, usuarios compartidos, cronología, unidades aumentadas y hashes. Las etiquetas se normalizan eliminando espacios exteriores, compactando espacios internos y comparando sin distinguir mayúsculas/minúsculas. El resultado siempre adopta una de las cuatro formas canónicas.

El reporte publicable se conserva en [`results/data_validation.json`](results/data_validation.json) sin texto de posts.

## 6. Anclas sintéticas

Las anclas representan las fronteras semánticas de cada clase. No son posts del corpus ni ejemplos supervisados. Se usaron dos estrategias:

1. `zero_shot`: expresiones directas y prototípicas derivadas de las definiciones;
2. `meta_prompting`: expresiones indirectas, coloquiales, dubitativas, fragmentarias, funcionales o contextualizadas.

Cada estrategia contiene 15 frases por clase:

```text
2 estrategias × 4 clases × 15 frases = 120 anclas
```

`Indicator` se trata como negativo difícil: puede reflejar malestar, pero excluye ideación, plan, preparación, acceso, autolesión e intento. Las demás clases preservan los límites de la tarea.

Los prompts son versión 1.0.0 y las anclas curadas versión 1.1.0:

| Archivo | SHA-256 |
|---|---|
| `zero_shot__risk_levels.json` | `9f1bd6c83faf81b15d291cde2690695dde7107cc9b03487626240c65971c4f10` |
| `meta_prompting__risk_levels.json` | `907dae22a50b2b15ead0096c749d71c7f5e076a8b9abbd618c39ad8e7f3fd267` |

El cargador consume únicamente `targets[*].phrases[*].text`, valida el orden de clases y rechaza etiquetas inesperadas o conjuntos vacíos. Se conservaron prompts, versiones, texto final y hashes. Los archivos no registran el proveedor ni la versión del modelo que generó las propuestas iniciales; esta carencia se declara como limitación de procedencia.

## 7. Representación semántica

Se utilizó `sentence-transformers/all-mpnet-base-v2`, revisión `e8c3b32edf5434bc2275fc9bab85f82640a19130`, con embeddings de 768 dimensiones, máximo de 384 tokens, batch de 32 y normalización L2. La ejecución registrada fue en CPU.

La similitud es el coseno entre vectores normalizados:

```text
cos(x, a) = (x / ||x||₂) · (a / ||a||₂)
```

### 7.1 Embedding global

Cada post completo se codifica una vez y produce 768 variables. Los posts que superan 384 tokens quedan truncados por el encoder.

### 7.2 Vista macro

Las 15 anclas de una clase se concatenan y codifican como un vector macro. Cada frase del post se compara con los cuatro vectores. Para cada clase se resumen las similitudes mediante media, mediana, desviación estándar, varianza, máximo, mínimo, percentil 25 y percentil 75. Una estrategia produce `4 × 8 = 32` variables macro.

### 7.3 Vista individual

Las 15 anclas se codifican por separado. Para cada frase se conserva la máxima similitud con las 15 anclas de una clase; después se aplican los mismos ocho estadísticos. Una estrategia produce 32 variables individuales.

## 8. Bloques de características

| Bloque | Dimensión |
|---|---:|
| `post_embedding` | 768 |
| `zero_shot__macro__sentence` | 32 |
| `zero_shot__individual__sentence` | 32 |
| `meta_prompting__macro__sentence` | 32 |
| `meta_prompting__individual__sentence` | 32 |
| **Total** | **896** |

Los archivos `.npz` almacenan IDs, grupos, nombres y matrices; train incluye además las etiquetas. Las cadenas usan Unicode fijo y se cargan con `allow_pickle=False`.

## 9. Estrategias de modelado del riesgo

### 9.1 Familias comparadas

La búsqueda examinó el embedding del post por sí solo, cada bloque de anclas, las vistas agrupadas por estrategia, la combinación macro `meta_prompting` con individual `zero_shot`, todas las anclas y dos fusiones del post con anclas. La fusión es temprana: los bloques se concatenan antes de ajustar un único modelo.

Se compararon regresión logística y SVM lineal, `C` en `{0.25, 1.0}` y selección ANOVA con `k` en `{64, 128, all}` cuando la dimensión lo permitía. `k=all` se excluyó de las fusiones tempranas para contener la complejidad. Quedaron 64 configuraciones válidas.

Cada configuración usa:

```text
StandardScaler → SelectKBest(f_classif, k) → clasificador multiclase
```

El escalador y ANOVA se ajustan dentro de cada fold. La regresión logística usa `lbfgs`, hasta 5,000 iteraciones y `class_weight="balanced"`. La SVM usa `LinearSVC`, `dual="auto"`, hasta 10,000 iteraciones y el mismo balanceo.

### 9.2 Validación agrupada

La validación empleó `StratifiedGroupKFold` con cinco folds, tres repeticiones, semillas 2026, 2027 y 2028, y `anon_user_id` como grupo. Se comprobó que ningún usuario apareciera en entrenamiento y validación del mismo fold.

```text
64 configuraciones × 5 folds × 3 repeticiones = 960 ajustes
```

La selección se hizo por el promedio del Risk Weighted F1 de los 15 folds. Macro F1 y accuracy fueron diagnósticos secundarios.

Después de elegir la configuración, cada post recibió tres predicciones OOF. La etiqueta agregada se decidió por mayoría; un empate entre tres clases distintas favoreció la clase menos severa según el orden canónico. Por esta agregación, el F1 OOF global no tiene que coincidir con la media de los folds.

## 10. Configuración seleccionada

| Elemento | Valor |
|---|---|
| Familia | `early_fusion_all` |
| Entrada | Cinco bloques, 896 variables |
| Escalado | `StandardScaler` |
| Selección | `SelectKBest(f_classif, k=128)` |
| Clasificador | Regresión logística balanceada |
| `C` | 0.25 |

Al ajustar el modelo final con todo train, las 128 variables seleccionadas se distribuyeron así:

| Procedencia | Seleccionadas |
|---|---:|
| Embedding del post | 59 |
| Zero-shot macro | 17 |
| Zero-shot individual | 18 |
| Meta-prompting macro | 17 |
| Meta-prompting individual | 17 |

La presencia de variables de los cinco bloques confirma que el modelo final consumió señal global y señal de anclas. No demuestra importancia causal.

### 10.1 Mejores variantes por familia

| Familia | Mejor configuración | Weighted F1 medio |
|---|---|---:|
| Post + todas las vistas | LR, `C=0.25`, `k=128` | **0.615877** |
| Post + vistas complementarias | LR, `C=0.25`, `k=128` | 0.613791 |
| Post MPNet | SVM, `C=0.25`, `k=128` | 0.608004 |
| Todas las anclas | LR, `C=0.25`, todas | 0.588822 |
| Anclas complementarias | LR, `C=0.25`, todas | 0.576171 |
| Anclas meta-prompting | SVM, `C=0.25`, todas | 0.575213 |
| Meta-prompting macro | LR, `C=0.25`, todas | 0.564564 |
| Anclas zero-shot | LR, `C=0.25`, todas | 0.562603 |
| Meta-prompting individual | SVM, `C=0.25`, todas | 0.550114 |
| Zero-shot macro | LR, `C=0.25`, todas | 0.547346 |
| Zero-shot individual | LR, `C=0.25`, todas | 0.542505 |

La ventaja media de la fusión completa fue 0.002086 frente a la fusión parcial y 0.007873 frente al post MPNet. Son diferencias pequeñas respecto a la variación entre folds.

## 11. Métricas de riesgo

Para una clase `c`:

```text
Precision_c = TP_c / (TP_c + FP_c)
Recall_c    = TP_c / (TP_c + FN_c)
F1_c        = 2 × Precision_c × Recall_c / (Precision_c + Recall_c)
```

Con `zero_division=0`, una división no definida produce cero.

Risk Weighted F1 es la métrica principal y pondera el F1 por soporte:

```text
Weighted F1 = Σ_c (n_c / N) × F1_c
```

Macro F1 da el mismo peso a cada clase:

```text
Macro F1 = (F1_Indicator + F1_Ideation + F1_Behavior + F1_Attempt) / 4
```

Accuracy es la proporción de aciertos. Se reportó como diagnóstico, no como criterio de selección.

## 12. Resultados de riesgo

### 12.1 Resultado OOF agregado

| Métrica | Valor |
|---|---:|
| Risk Weighted F1 | **0.6257467580** |
| Risk Macro F1 | 0.5511300392 |
| Accuracy | 0.6214067278 |

### 12.2 Por clase

| Clase | Precision | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| Indicator | 0.759936 | 0.782324 | **0.770968** | 611 |
| Ideation | 0.620390 | 0.551060 | **0.583673** | 519 |
| Behavior | 0.568306 | 0.531969 | **0.549538** | 391 |
| Attempt | 0.245810 | 0.385965 | **0.300341** | 114 |

`Indicator` obtuvo el mejor F1. `Attempt` tuvo la menor precisión y el menor F1; el modelo recuperó 44 de 114 intentos, pero generó numerosos falsos positivos.

### 12.3 Matriz de confusión

Las filas son gold y las columnas predicción.

| Gold \ Predicción | Indicator | Ideation | Behavior | Attempt |
|---|---:|---:|---:|---:|
| Indicator | 478 | 65 | 35 | 33 |
| Ideation | 87 | 286 | 93 | 53 |
| Behavior | 47 | 87 | 208 | 49 |
| Attempt | 17 | 23 | 30 | 44 |

`Ideation` se confundió con `Behavior` en 93 casos y `Behavior` con `Ideation` en 87. De los 70 intentos no detectados, 30 se asignaron a `Behavior`, 23 a `Ideation` y 17 a `Indicator`.

### 12.4 Estabilidad por fold

| Estadístico, 15 folds | Weighted F1 | Macro F1 | Accuracy |
|---|---:|---:|---:|
| Media | 0.615877 | 0.533871 | 0.607177 |
| Desviación estándar muestral | 0.020992 | 0.032527 | 0.022486 |
| Mínimo | 0.585000 | 0.473819 | 0.566474 |
| Mediana | 0.613713 | 0.533535 | 0.600639 |
| Máximo | 0.656295 | 0.598221 | 0.647416 |

La amplitud observada del Weighted F1 fue 0.071296, lo que muestra sensibilidad a la composición de usuarios de cada fold.

## 13. Extracción de evidencia

### 13.1 Dependencia del riesgo

El extractor recibe la clase predicha. Si es `Indicator`, no genera candidatos y devuelve evidencia vacía. Para las demás clases solo usa anclas de la etiqueta predicha. Esto mantiene coherencia entre salidas, pero propaga los errores del clasificador.

### 13.2 Candidatos con offsets

Los candidatos se generan desde el `post` original mediante:

1. oraciones separadas tras `.`, `!`, `?` o salto de línea;
2. cláusulas separadas por punto y coma, coma no decimal, raya o semirraya;
3. ventanas de seis tokens con stride de tres;
4. una ventana final alineada al término del post cuando es necesaria.

Se conservan candidatos de 1 a 40 tokens, se eliminan los que contienen `;`, se deduplican por `(start, end)` y se comprueba `post[start:end] == span.text`.

Con las clases OOF se generaron 35,993 candidatos. Hubo al menos uno compatible para 1,380 de 1,833 spans gold: cobertura enrutada de 75.29 %. El diagnóstico incluye los errores de riesgo, ya que una fila predicha como `Indicator` no genera candidatos.

### 13.3 Scoring y selección

Cada candidato se codifica con MPNet. Para la clase predicha se obtiene el máximo coseno con las 15 anclas de cada estrategia y se promedian ambos máximos:

```text
score(span, clase) =
  [max cos(span, anclas_zero_shot_clase)
   + max cos(span, anclas_meta_prompting_clase)] / 2
```

Los spans se ordenan por score descendente, menor longitud y posición inicial. La búsqueda comparó 60 combinaciones:

```text
top_k ∈ {1, 2, 3}
min_similarity ∈ {0.20, 0.25, 0.30, 0.35, 0.40}
max_tokens ∈ {12, 20, 32, 40}
```

Se rechaza un span cuando su intersección sobre unión con otro ya elegido supera 0.8. La configuración final fue `top_k=2`, similitud mínima 0.40 y máximo de 12 tokens. El umbral 0.35 dio el mismo F1; el desempate determinista eligió el umbral más restrictivo.

## 14. Phrase F1

Predicción y gold se convierten con `casefold` y se compactan los espacios. Una pareja coincide si una frase contiene a la otra y la predicción no supera tres veces la longitud en tokens del gold:

```text
predicción ⊆ gold  o  gold ⊆ predicción
tokens(predicción) ≤ 3 × tokens(gold)
```

Cada predicción y cada gold participan como máximo en una pareja. La implementación construye el grafo de compatibilidad y calcula un emparejamiento máximo.

Para cada post:

```text
PhrasePrecision = emparejamientos / spans predichos
PhraseRecall    = emparejamientos / spans gold
PhraseF1        = 2 × Precision × Recall / (Precision + Recall)
```

El Phrase F1 final es la media de los F1 por post. La implementación local asigna 1.0 cuando predicción y gold están vacíos. La definición proporcionada no especifica expresamente ese caso; debe confirmarse con el scorer oficial.

## 15. Resultados de evidencia

| Diagnóstico | Resultado |
|---|---:|
| Evidence Phrase F1 | **0.493796** |
| F1 en posts con gold no vacío | 0.296425 |
| F1 en posts con gold vacío | 0.845238 |
| Filas OOF con evidencia | 899 |
| Spans OOF predichos | 1,685 |
| Spans OOF no verbatim | 0 |

El contraste entre gold vacío y no vacío muestra que una parte importante del score proviene de acertar salidas vacías. El F1 de 0.296425 describe mejor la recuperación cuando sí existe evidencia.

| Clase gold | Phrase F1 medio | Posts |
|---|---:|---:|
| Indicator | 0.818525 | 611 |
| Ideation | 0.367154 | 519 |
| Behavior | 0.237994 | 391 |
| Attempt | 0.207270 | 114 |

Estos valores por clase son diagnósticos locales, no componentes oficiales separados.

## 16. Submission

El modelo final se ajustó con las 1,635 filas y se aplicó al test. La evidencia se recuperó usando sus clases predichas.

La estructura del archivo privado `outputs/subtask1_post_frases/PostFrasesV1.csv` se resume, sin evidencia textual, en [`results/submission_validation.json`](results/submission_validation.json).

| Control | Resultado |
|---|---:|
| Filas / `row_id` únicos | 378 / 378 |
| Duplicados | 0 |
| Filas con evidencia | 208 |
| Spans de evidencia | 386 |
| Spans no verbatim | 0 |
| Evidencia no vacía para `Indicator` | 0 |
| Filas con factores | 0 |
| Tamaño | 20,521 bytes |

| Clase predicha | Filas |
|---|---:|
| Indicator | 134 |
| Ideation | 94 |
| Behavior | 89 |
| Attempt | 61 |

SHA-256:

```text
55dd4462889b03075ad43c01e380e56ceecb7a4a12546210f159b1d2950092ae
```

Test no contiene etiquetas accesibles. Su distribución y sus conteos son controles estructurales, no medidas de calidad.

## 17. Artefactos y trazabilidad

| Artefacto | Contenido |
|---|---|
| `data_validation.json` | Conteos, hashes y controles sin texto. |
| `split_manifest.json` | Filas y usuarios de los 15 folds. |
| `feature_manifest.json` | Modelo, revisión, configuración, hashes y entorno. |
| `train_features.npz` | Bloques de train, etiquetas, IDs y grupos. |
| `test_features.npz` | Bloques de test, IDs y grupos. |
| `risk_cv_fold_metrics.csv` | Métricas de los 960 ajustes. |
| `risk_cv_summary.csv` | Media y desviación por configuración. |
| `oof_risk_predictions.csv` | Gold y predicción OOF por fila. |
| `risk_oof_report.json` | Modelo elegido, métricas y matriz de confusión. |
| `evidence_tuning.json` | Resultados de las 60 combinaciones de evidencia. |
| `oof_evidence_predictions.csv` | Evidencia OOF por fila. |
| `subtask1_oof_report.json` | Resultado conjunto de la Tarea 1. |
| `risk_model.joblib` | Pipeline final y nombres de variables. |
| `PostFrasesV1.csv` | Submission de 378 filas. |

Los CSV de evidencia, el modelo y la submission pueden contener texto o vocabulario derivado del corpus restringido. Deben permanecer privados.

## 18. Organización del código

### 18.1 Scripts

| Archivo | Responsabilidad |
|---|---|
| `01_validate_data.py` | Validar workbooks, usuarios, cronología y frases. |
| `02_prepare_user_splits.py` | Crear folds agrupados por usuario. |
| `03_build_features.py` | Codificar posts, frases y anclas. |
| `04_train_risk_models.py` | Comparar configuraciones, crear OOF y ajustar el modelo. |
| `05_tune_evidence.py` | Puntuar candidatos y seleccionar hiperparámetros. |
| `06_predict_submission.py` | Predecir test y validar el CSV. |
| `07_evaluate.py` | Calcular las métricas OOF conjuntas. |

### 18.2 Módulos

| Módulo | Responsabilidad |
|---|---|
| `anchors.py` | Carga estricta de anclas. |
| `config.py` | Configuración, rutas, semillas y hashes. |
| `constants.py` | Etiquetas y esquema canónicos. |
| `data.py` | Lectura, normalización y parsing seguro. |
| `embedding.py` | MPNet y similitud coseno. |
| `features.py` | Construcción de los cinco bloques. |
| `modeling.py` | Splits, búsqueda, modelos y métricas. |
| `spans.py` | Candidatos con offsets. |
| `evidence.py` | Ranking, Phrase F1 y validación verbatim. |

## 19. Reproducción

Desde la raíz de la carpeta aislada:

```bash
python -m pip install -e '.[dev]'
python scripts/01_validate_data.py
python scripts/02_prepare_user_splits.py
python scripts/03_build_features.py
python scripts/04_train_risk_models.py
python scripts/05_tune_evidence.py
python scripts/06_predict_submission.py --team-name PostFrasesV1
python scripts/07_evaluate.py
python -m pytest
ruff check .
```

Las fases de embeddings requieren la revisión fijada de MPNet. La caché se guarda en `.hf-cache` dentro de la carpeta, salvo que `HF_HOME` ya esté definido.

### 19.1 Entorno registrado

| Componente | Versión |
|---|---|
| Python | 3.12.11 |
| NumPy | 2.2.6 |
| pandas | 2.3.2 |
| scikit-learn | 1.7.2 |
| sentence-transformers | 5.1.1 |
| PyTorch | 2.8.0+cpu |
| openpyxl | 3.1.5 |
| PyYAML | 6.0.3 |
| joblib | 1.5.2 |
| tqdm | 4.67.1 |

La semilla general fue 2026. La ejecución registrada se realizó en Linux sobre CPU.

## 20. Pruebas automatizadas

Las 15 pruebas cubren carga de anclas, etiquetas, parsing seguro, bloques de características, persistencia, grupos sin fuga, generación de spans, offsets, contención sin distinguir mayúsculas/minúsculas, límite de longitud tres a uno, emparejamiento uno a uno, selección verbatim y uso de `post_frases`.

En la verificación del 29 de agosto de 2026 pasaron las 15 pruebas y Ruff no encontró errores.

## 21. Decisiones metodológicas

### Frases suministradas para riesgo

Las frases aumentadas están disponibles y completamente alineadas en ambas particiones. Utilizarlas evita introducir una segmentación distinta entre train y test.

### Post original para evidencia

La extracción vuelve siempre al `post`. Así se aprovecha `post_frases` para el riesgo sin perder la garantía de que cada evidencia es una subcadena literal con offsets recuperables.

### Fusión temprana

La concatenación permite que ANOVA y la regresión logística ponderen conjuntamente la semántica global y las similitudes con anclas. La búsqueda eligió los cinco bloques.

### Validación por usuario

Agrupar por `anon_user_id` evita que posts del mismo autor aparezcan en entrenamiento y validación. Sin esta restricción, el score podría aprovechar regularidades de autor.

### Balanceo de clases

Los pesos inversos a la frecuencia compensan parcialmente la escasez de `Attempt`. Su F1 de 0.300341 muestra que el balanceo no resolvió la frontera con `Behavior` e `Ideation`.

### Evidencia condicionada

La clase predicha restringe el conjunto de anclas. El mecanismo aporta coherencia, pero acopla los errores de ambos componentes.

## 22. Limitaciones

1. **Sin gold del test.** Solo existen métricas OOF y controles de estructura.
2. **Selección y reporte sobre OOF.** Falta un holdout intacto o validación anidada.
3. **153 usuarios.** La composición de folds produce variación apreciable.
4. **Desbalance.** `Attempt` tiene 114 posts y F1 0.300341.
5. **Convención empty-empty.** El Phrase F1 local asigna 1.0; falta verificar el scorer oficial.
6. **Gold no alineable.** 173 de 1,833 spans no aparecen literalmente en el post.
7. **Cobertura.** El 24.71 % de los spans gold no tuvo candidato compatible bajo el enrutamiento OOF.
8. **Propagación del riesgo.** Una clase incorrecta cambia las anclas o fuerza evidencia vacía.
9. **Truncamiento.** MPNet limita las entradas a 384 tokens.
10. **Procedencia incompleta.** No se registró el proveedor/modelo que produjo las propuestas iniciales de anclas.
11. **Generalización.** Plataforma, vocabulario, época y población pueden cambiar fuera del corpus.
12. **Uso no clínico.** Las salidas no son diagnóstico ni sustituyen una evaluación profesional.

## 23. Privacidad y seguridad

Los workbooks crudos no se modifican y esta documentación no reproduce posts. Las matrices, predicciones, evidencias y modelos derivados deben tratarse como artefactos restringidos.

Las anclas emplean lenguaje genérico y no accionable: excluyen métodos nombrados, cantidades, instrucciones, ubicaciones y descripciones gráficas. El sistema se diseñó para investigación y evaluación en la competición, no para intervención autónoma.

## 24. Qué puede concluirse

La fusión de un embedding global con cuatro vistas de similitud produjo el mejor promedio dentro de la búsqueda ejecutada. El resultado respalda el uso conjunto de ambas señales en este protocolo OOF, pero la diferencia frente a las alternativas cercanas es pequeña y no establece superioridad fuera de estos folds.

La clasificación funcionó mejor para `Indicator` y peor para `Attempt`. En evidencia, todos los spans fueron literales, pero el F1 de los posts con gold no vacío fue 0.296425. La principal dificultad del extractor fue recuperar evidencia positiva, especialmente en `Behavior` y `Attempt`.

## 25. Cierre

`PostFrasesV1` utiliza cinco bloques semánticos, selección ANOVA y regresión logística balanceada para clasificar riesgo. Después recupera hasta dos spans de máximo 12 tokens mediante similitud con anclas de la clase predicha. El sistema alcanzó 0.625747 de Risk Weighted F1, 0.493796 de Phrase F1 y 0.569196 como score normalizado de la Tarea 1.

La submission contiene las 378 filas requeridas y supera los controles estructurales implementados. Su desempeño real sobre test no puede determinarse sin las etiquetas ocultas o una evaluación oficial.
