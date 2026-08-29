# Análisis completo de resultados — Subtask 1

## 1. Alcance del análisis

Este documento analiza exclusivamente el sistema `PostFrasesV1` para la
Subtask 1: clasificación del riesgo suicida en `Indicator`, `Ideation`,
`Behavior` y `Attempt`, junto con la extracción de evidencia textual. No se
modelan los 24 factores de la Subtask 2.

Los resultados presentados son exploratorios y proceden de validación cruzada
OOF (*out-of-fold*) sobre el conjunto de entrenamiento. No son resultados del
test oculto ni puntuaciones del leaderboard.

## 2. Qué datos se usan para entrenar, validar y predecir

| Etapa | Archivo | Etiquetas disponibles | Uso |
|---|---|---:|---|
| Entrenamiento y validación OOF | `train dataset-con-frases-con-segmentosV1.xlsx` | Sí | Selección de configuración, cálculo de métricas y ajuste final |
| Inferencia de competencia | `test dataset-con-frases.xlsx` | No | Producción de riesgo y evidencia para 378 publicaciones |

El conjunto de entrenamiento contiene 1,635 publicaciones de 153 usuarios. El
test contiene 378 publicaciones de 36 usuarios. No hay usuarios compartidos
entre ambos conjuntos.

Por tanto, el flujo correcto es:

1. Las configuraciones se comparan mediante cinco particiones agrupadas por
   `anon_user_id`, repetidas tres veces.
2. Cada publicación de train recibe una predicción OOF en cada repetición sin
   haber sido utilizada para ajustar el modelo que la predice.
3. La configuración se selecciona por el promedio del Weighted F1 de riesgo en
   los 15 folds.
4. Para el informe agregado, las tres predicciones OOF de cada publicación se
   combinan por voto mayoritario.
5. La configuración seleccionada se vuelve a entrenar con las 1,635 filas de
   train.
6. Ese modelo final genera las predicciones de `test dataset-con-frases.xlsx`.
7. Como el test no tiene riesgo ni evidencia de referencia, no es posible
   calcular localmente Weighted F1 ni Phrase F1 sobre sus 378 filas. Esa
   evaluación solo puede realizarla el servidor con las etiquetas ocultas.

La columna `post_frases` se consume tanto en train como en test para construir
las características de similitud semántica. Las 9,276 frases de train y las
2,395 frases de test presentan alineación literal del 100% con su publicación
de origen. Las columnas `post_segmentos` y `post_segmentosV1` no se usan porque
no existe una representación equivalente y auditada en el test.

La evidencia final se extrae siempre desde el campo original `post`, no desde
`post_frases`. Esto garantiza que cada span entregado sea una subcadena literal
de la publicación y pueda validarse antes de crear la entrega.

## 3. Distribución y dificultad de los datos

| Clase | Casos | Proporción |
|---|---:|---:|
| Indicator | 611 | 37.37% |
| Ideation | 519 | 31.74% |
| Behavior | 391 | 23.91% |
| Attempt | 114 | 6.97% |
| Total | 1,635 | 100% |

La distribución es claramente desbalanceada. `Attempt` tiene aproximadamente
una quinta parte de los ejemplos de `Indicator`. Por ello, el Weighted F1 puede
verse razonablemente alto aunque el desempeño de `Attempt` sea bajo. El Macro
F1 y las métricas por clase son indispensables para interpretar el sistema.

## 4. Espacio experimental

Se evaluaron 64 configuraciones, cada una en 15 folds, para un total de 960
ajustes de validación. Las variantes combinaron:

- embedding completo de la publicación, de 768 dimensiones;
- anclas `zero_shot` y `meta_prompting`;
- representación macro e individual sobre `post_frases`;
- regresión logística y SVM lineal;
- regularización `C` en 0.25 y 1.0;
- selección ANOVA fold-local de 64 o 128 variables, o todas cuando la
  configuración lo permitía.

Cada bloque de anclas tiene 32 variables: ocho estadísticos para cada una de
las cuatro clases. La fusión completa contiene 896 variables de entrada: 768
del post y cuatro bloques de 32 variables. La configuración ganadora retiene
128 variables dentro de cada fold; tanto el escalado como la selección se
ajustan únicamente con la parte de entrenamiento del fold.

## 5. Configuración seleccionada

La mejor configuración fue:

- experimento: `early_fusion_all`;
- bloques: embedding del post y las cuatro vistas de anclas;
- clasificador: regresión logística balanceada;
- `C = 0.25`;
- selección fold-local: 128 de 896 variables.

Su promedio sobre los 15 folds fue:

| Métrica | Media | Desviación estándar | Mínimo | Máximo |
|---|---:|---:|---:|---:|
| Risk Weighted F1 | 0.615877 | 0.020992 | 0.585000 | 0.656295 |
| Risk Macro F1 | 0.533871 | 0.032527 | 0.473819 | 0.598221 |
| Accuracy | 0.607177 | 0.022486 | 0.566474 | 0.647416 |

El intervalo descriptivo media ± 1.96 desviaciones estándar para Weighted F1
es aproximadamente 0.5747–0.6570. Un intervalo aproximado del 95% para la media
de los 15 folds sería 0.604–0.628, pero debe interpretarse con cautela: los folds
de validación repetida no son observaciones completamente independientes.

La desviación de 0.021 y el rango observado indican variación moderada entre
grupos de usuarios. La configuración es relativamente estable, aunque no debe
presentarse como una estimación definitiva de generalización al test oculto.

## 6. Comparación de configuraciones

Las primeras configuraciones por Weighted F1 medio fueron:

| Posición | Configuración | Clasificador | C | k | Weighted F1 medio | Desv. |
|---:|---|---|---:|---:|---:|---:|
| 1 | Fusión temprana completa | Regresión logística | 0.25 | 128 | 0.615877 | 0.020992 |
| 2 | Fusión temprana complementaria | Regresión logística | 0.25 | 128 | 0.613791 | 0.033959 |
| 3 | Fusión temprana complementaria | SVM lineal | 0.25 | 128 | 0.613086 | 0.035170 |
| 4 | Fusión temprana complementaria | SVM lineal | 1.00 | 128 | 0.610058 | 0.034432 |
| 5 | Fusión temprana complementaria | Regresión logística | 1.00 | 128 | 0.609348 | 0.032292 |
| 6 | Fusión temprana completa | Regresión logística | 1.00 | 128 | 0.608195 | 0.020347 |
| 7 | Solo embedding del post | SVM lineal | 0.25 | 128 | 0.608004 | 0.020716 |

La ganancia de la configuración seleccionada frente al mejor modelo que usa
solo el embedding del post es 0.007873 de Weighted F1 medio. Es una mejora
pequeña, pero consistente con la hipótesis de que las comparaciones entre
frases y anclas aportan información complementaria. No demuestra por sí sola
una mejora estadísticamente concluyente, porque no se realizó una prueba
pareada corregida para validación cruzada repetida.

La diferencia entre el primer y segundo lugar es únicamente 0.002086. La fusión
completa, sin embargo, tiene menor variabilidad entre folds: 0.020992 frente a
0.033959. La elección se justifica tanto por el mejor promedio como por esa
mayor estabilidad.

Al agrupar las configuraciones por familia, los mejores resultados aparecen
cuando el embedding del post y las anclas se aprenden conjuntamente. El mejor
modelo solo con anclas alcanza 0.588822, mientras que la fusión completa llega
a 0.615877. Esto muestra que las anclas funcionan mejor como complemento del
contenido semántico general que como sustituto del post.

La selección de 128 variables fue claramente favorable: las 12
configuraciones con `k=128` promedian 0.607445, frente a 0.592610 para `k=64` y
0.558592 para las configuraciones sin selección. La reducción supervisada
elimina ruido y redundancia de los 896 atributos. `C=0.25` también supera en
promedio a `C=1.0` (0.578311 frente a 0.574202), lo que sugiere que una
regularización más fuerte mejora la generalización.

## 7. Resultado OOF agregado de riesgo

Después de seleccionar la configuración, cada fila recibió tres predicciones
OOF, una por repetición. El voto mayoritario produce:

| Métrica | Resultado agregado |
|---|---:|
| Risk Weighted F1 | **0.625747** |
| Risk Macro F1 | **0.551130** |
| Accuracy | **0.621407** |

Este Weighted F1 agregado es mayor que la media de folds de 0.615877 porque el
voto mayoritario de tres repeticiones corrige algunos errores inestables. No
son dos experimentos distintos: uno es el criterio de selección promedio y el
otro resume una predicción OOF consolidada por publicación.

## 8. Análisis por clase

| Clase | Precisión | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| Indicator | 0.759936 | 0.782324 | **0.770968** | 611 |
| Ideation | 0.620390 | 0.551060 | **0.583673** | 519 |
| Behavior | 0.568306 | 0.531969 | **0.549538** | 391 |
| Attempt | 0.245810 | 0.385965 | **0.300341** | 114 |

`Indicator` es la clase mejor resuelta. Su alta precisión y recall indican que
el modelo identifica bien muchas publicaciones sin mención suicida explícita.
`Ideation` y `Behavior` tienen un desempeño intermedio y parecido, pero la
frontera entre expresión suicida y existencia de plan o conducta sigue siendo
difícil.

`Attempt` es el principal punto débil. Su recall de 0.386 significa que solo 44
de 114 intentos se identifican correctamente. Su precisión de 0.246 indica que
el modelo predice `Attempt` con exceso: genera 179 predicciones para solo 114
casos reales. La ponderación de clases ayuda a recuperar más casos raros, pero
produce numerosos falsos positivos.

## 9. Matriz de confusión

Las filas son clases reales y las columnas, clases predichas:

| Real \ Predicha | Indicator | Ideation | Behavior | Attempt |
|---|---:|---:|---:|---:|
| Indicator | **478** | 65 | 35 | 33 |
| Ideation | 87 | **286** | 93 | 53 |
| Behavior | 47 | 87 | **208** | 49 |
| Attempt | 17 | 23 | 30 | **44** |

Los dos errores más frecuentes son `Ideation → Behavior` y
`Behavior → Ideation`, con 93 y 87 casos. Esta simetría confirma que la frontera
central de la taxonomía —expresión suicida sin plan frente a expresión con
plan o conducta— es el principal problema de separación semántica.

De los casos reales de `Attempt`, 26.3% se clasifican como `Behavior`, 20.2%
como `Ideation` y 14.9% como `Indicator`. Los errores hacia `Behavior` pueden
reflejar dificultad para distinguir una conducta o plan presente de la mención
de un intento pasado o reciente. Los 17 casos enviados a `Indicator` son los
errores más graves desde la perspectiva de sensibilidad, aunque este sistema
es experimental y no debe interpretarse como herramienta clínica.

También aparecen 33 casos `Indicator → Attempt`. Esto, junto con el exceso de
predicciones de `Attempt`, indica que ciertas expresiones generales de crisis,
temporalidad o primera persona pueden activar demasiado las anclas de mayor
riesgo.

## 10. Extracción de evidencia

El mejor resultado OOF de evidencia fue:

| Parámetro | Valor |
|---|---:|
| Número máximo de spans | 2 |
| Similitud mínima | 0.40 |
| Longitud máxima | 12 tokens |
| Phrase F1 | **0.493796** |

El sistema genera candidatos como oraciones, cláusulas y ventanas de seis
tokens sobre el `post` intacto. Cada candidato se compara mediante MPNet con
las anclas de la clase de riesgo predicha. Para `Indicator` se emite evidencia
vacía.

Los resultados de la rejilla muestran tres patrones claros:

- dos spans superan en promedio a uno o tres: 0.488020 frente a 0.482935 y
  0.479777;
- el límite de 12 tokens es el mejor en promedio, con 0.488859;
- permitir 20, 32 o 40 tokens reduce progresivamente el promedio a 0.484354,
  0.481295 y 0.479800.

La preferencia por spans cortos es coherente con la métrica oficial: una
predicción demasiado larga puede ser penalizada si supera tres veces la
longitud del span de referencia. Emitir tres spans también reduce precisión al
introducir candidatos secundarios.

El umbral 0.40 obtiene el mejor promedio entre umbrales, aunque 0.35 alcanza el
mismo máximo en la combinación ganadora. Esto sugiere una pequeña meseta local,
no un punto extremadamente sensible. Aun así, los parámetros de evidencia se
ajustaron sobre las mismas predicciones OOF empleadas para informar el
resultado; por ello, 0.493796 puede tener optimismo de selección.

Además, Phrase F1 depende de la clase de riesgo predicha: si el clasificador
elige la clase incorrecta, el recuperador consulta anclas incorrectas. Mejorar
la clasificación, especialmente en `Ideation`, `Behavior` y `Attempt`, puede
mejorar simultáneamente la evidencia.

La implementación local asigna F1=1 cuando tanto la evidencia predicha como la
evidencia de referencia están vacías. Esta convención debe verificarse contra
el evaluador oficial antes de considerar el valor estrictamente comparable con
el leaderboard.

## 11. Resultado conjunto de la Subtask 1

La Subtask 1 aporta 70% de la puntuación total: 40% por clasificación de riesgo
y 30% por evidencia.

| Componente | Métrica OOF | Peso completo | Contribución |
|---|---:|---:|---:|
| Riesgo | 0.625747 | 0.40 | 0.250299 |
| Evidencia | 0.493796 | 0.30 | 0.148139 |
| Total Subtask 1 | — | 0.70 | **0.398438** |

Normalizando únicamente dentro del peso de la Subtask 1:

`0.398438 / 0.70 = 0.569196`.

La cifra 0.569196 es la síntesis más útil cuando se analiza solo esta subtarea.
No debe confundirse con la puntuación completa de la competencia. Como la
Subtask 2 no se modela, el placeholder local de Factor Macro F1 es cero.

## 12. Predicciones sobre el test con frases

El modelo final se ajustó con todas las filas etiquetadas de train y se aplicó
a las 378 filas de `test dataset-con-frases.xlsx`. La entrega contiene:

| Diagnóstico de entrega | Resultado |
|---|---:|
| Filas | 378 |
| `row_id` únicos | 378 |
| Indicator | 134 |
| Ideation | 94 |
| Behavior | 89 |
| Attempt | 61 |
| Filas con evidencia | 208 |
| Spans de evidencia | 386 |
| Spans no literales | 0 |
| Indicator con evidencia no vacía | 0 |

Estas cifras validan formato, cobertura y restricciones de evidencia, pero no
miden calidad predictiva. La distribución del test no se utilizó para elegir
el modelo ni ajustar umbrales.

La distribución predicha de `Attempt` es 16.1% en test, frente a 7.0% de casos
reales en train. El OOF también mostró sobrepredicción de `Attempt` (179
predicciones frente a 114 casos). Esto es una señal de posible descalibración,
pero no autoriza a modificar predicciones usando la distribución del test: sin
etiquetas, hacerlo sería una corrección especulativa.

## 13. Fortalezas

- separación de usuarios en todos los folds, evitando que publicaciones de la
  misma persona aparezcan simultáneamente en entrenamiento y validación;
- ausencia de usuarios compartidos entre train y test;
- alineación literal del 100% de `post_frases` en ambos archivos;
- selección de variables y escalado dentro de cada fold;
- combinación de semántica global del post con detectores basados en anclas;
- extracción de evidencia desde el texto original con offsets verificables;
- 386 spans de test validados como literales y ninguna evidencia para
  `Indicator`;
- configuración, semillas, versión de MPNet y dependencias registradas.

## 14. Limitaciones

- no existe una evaluación local genuina sobre el test porque sus etiquetas
  son ocultas;
- la configuración y los parámetros de evidencia se eligieron con el mismo
  esquema OOF usado para reportar resultados, lo que introduce optimismo de
  selección;
- solo hay 114 casos `Attempt`, y su F1 de 0.300341 es insuficiente para afirmar
  una separación robusta;
- la diferencia entre las mejores configuraciones es pequeña respecto de la
  variabilidad entre folds;
- no se han calculado intervalos de confianza corregidos ni pruebas pareadas
  apropiadas para validación cruzada repetida;
- el Phrase F1 local conserva una convención de casos vacío/vacío aún no
  contrastada con el scorer oficial;
- las anclas se usan como señales semánticas y no garantizan una explicación
  causal o clínica;
- el sistema no modela la Subtask 2 y no debe describirse como sistema clínico,
  diagnóstico o de intervención autónoma.

## 15. Conclusión

El resultado más sólido del experimento es la fusión temprana de MPNet del post
con las cuatro vistas de similitud entre `post_frases` y anclas. Alcanza
Weighted F1 OOF agregado de 0.625747 y Phrase F1 de 0.493796, para una
puntuación normalizada de Subtask 1 de 0.569196.

La fusión mejora modestamente al mejor modelo post-only y presenta mejor
estabilidad que la alternativa complementaria. El sistema funciona mejor para
`Indicator`, mantiene desempeño intermedio en `Ideation` y `Behavior`, y falla
principalmente en `Attempt`. La evidencia se beneficia de spans breves y de
emitir como máximo dos candidatos.

El test con frases sí se utiliza, pero únicamente para generar la entrega. La
calidad sobre ese conjunto solo será conocida cuando el servidor de la
competencia compare las predicciones con las etiquetas ocultas.
