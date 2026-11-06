# ConceptFAN: объединённая исследовательская постановка и проверка границ семантики

**Редакция: 6 ноября 2026. Состояние: research manuscript + интеграционный runtime; не публикационная заявка о SOTA.**

## Основная научная идея

Предсказательная полезность, вычислительное использование и физическая идентификация концепта — три разных свойства. Корректная система должна: (1) вычислять прогноз **только** по явно представленным концептам; (2) точно восстанавливать вклад каждого концепта; (3) разрешать локальные экспертные вмешательства; (4) отказывать в утверждении о физическом происхождении без независимого измерительного якоря; (5) сохранять исходный безопасный fallback при отказе.

Отсюда научно-исследовательский вопрос: *когда возможно физически именованное концептное представление и когда его необходимо честно оставлять статистическим прокси, даже если AUPRC высока?*

### Фактически проверенные результаты (НЕ один общий TEST)

| Исследование | Свои новые synthetic devices | Первичная проверка | Подтверждённые наблюдения |
|---|---|---|---|
| [V47 PR #44](https://github.com/fims9000/FuzzyAttentionNetworks/pull/44), [CI 42253997809](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/42253997809) | 4, TRAIN1000 + VAL400+400 + TEST600 | native Transformer, strict signed concept head | healthy MCAR40: K2 AP .68219; frozen V44 .74368; jointly V47 .74516; HGB1800 .73047. V47−V44 ≈.00148, **нет доказанного преимущества** |
| [V48 PR #45](https://github.com/fims9000/FuzzyAttentionNetworks/pull/45), [CI 42255848482](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/42255848482) | другие 4 | monotone bounded log-odds q→risk | healthy V48 AP .74483, но неизвестный OFF drift AP .63709 и NLL .80381 против K2 NLL .61938; усиление неверного q опасно |
| [V49 PR #46](https://github.com/fims9000/FuzzyAttentionNetworks/pull/46), [CI 42259872589](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/42259872589) | следующие 4 | equal-budget ideal synthetic expert 10/20% | при OFF drift active20 NLL .70014 vs unreviewed .79894, но K2 .61429; active20−random20 AP CIs пересекают 0 на всех 4 приборах |
| [V50 PR #47](https://github.com/fims9000/FuzzyAttentionNetworks/pull/47), [CI 42263861049](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/42263861049) | 4 × 2 orthogonal gauges | exact observational twins and external assay | то же наблюдаемое X/Y, разные именованные латентные E1; среднее |E1−E1'| .40806, blind MAE lower bound .20403. При TRAIN assay60 + дополнительно VAL160 energy Pearson .86363, MAE .37938 |
| [V51 PR #48](https://github.com/fims9000/FuzzyAttentionNetworks/pull/48), [CI 42265205302](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/42265205302) | ещё 4 × 2 gauges | TRAIN-only physical assay energy debias | при TRAIN assay240 + VAL160, MCAR40 energy MAE V50 raw .39065→V51 affine .34354 (12.1%); **качество итогового риска не менялось** |

У V47–V49 и V50–V51 **разные** синтетические генераторы и разные TEST, нельзя арифметически объединять модели и говорить, будто новое end-to-end качество уже измерено. HGB1800 из части исследований имеет **1800 бинарных labels**, концептная модель — 1000 binary TRAIN labels плюс отдельные VAL: бюджеты обучения различаются. V49 внешний oracle использует скрытую физику симулятора, чего нет в реальном применении.

## Архитектурная сборка

[Интеграционный PR #49](https://github.com/fims9000/FuzzyAttentionNetworks/pull/49) соединяет строго additive V47 model, источник энергии на основе независимого TRAIN/VALID физического assay (V50/V51), V49 intervention и exact K2 fallback. Внешняя source mapping q определяется **только** купленным TRAIN assay, а не TEST truth. Встроенная проверка \`ConceptRiskTrainingContractV52\` запрещает использовать V47 головку, обученную на V44 statistic-q, с новым q от V51 physical energy без отдельного совместного TRAIN обучения: иначе это подмена статистического признака, а не прогресс модели.

- \`src/conceptfan_discovery/conceptfan_integrated_runtime_v52.py\`: physical source receipt, device-gauge validation, q TRAIN normalizer, native signed concepts, exact K2 fallback, V49 verified expert actions.
- \`tests/discovery/test_conceptfan_integrated_runtime_v52.py\`: реально проверенные измерительные калибровки V51, torch additive risk, unknown hardware x1.5 drift, пропущенные значения и совпадение external reference ID.
- [CI 42268658215](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/42268658215): **25 tests passed**. Это **integration testing**, не TRAIN/VALID/TEST proof предсказательного преимущества V52.

## Математический необходимый отрицательный контроль

Для источников \(s\), двух матриц наблюдения \(H_A,H_B\), поворота \(R\) и соответствующей новой модели \(s'=Rs,\ H'=HR^\top\):
\[
X'=H's'+\varepsilon = X,\quad Y'=Y,\quad E'_1\neq E_1.
\]
Для любого физически слепого \(\hat E_1(X,Y)\):
\[
\frac{\mathrm{MAE}(\hat E_1,E_1)+\mathrm{MAE}(\hat E_1,E'_1)}{2}
\ge \frac{\mathbb E|E_1-E'_1|}{2}.
\]
Это стандартная наблюдательная неидентифицируемость (симметрия латентного пространства), **не новая теорема**. Внешний source-specific assay нарушает симметрию только для данного физического прибора; изменение gauge при точно сохранённом X остаётся невидимым для внутреннего anomaly monitor.

## Что требуется для полноценной научной публикации о новом end-to-end методе

**Новый preregistered TEST**, новая физическая семья приборов, у всех моделей один бюджет Y, а у семантически обучаемых контролей — одинаковый бюджет независимых физических assay:
1. K2 baseline; native V47 jointly с V44-statistical q; native V47 jointly с V51-anchored energy q; HGB(X); HGB(X, calibrated q), все TRAIN1000.
2. TEST AUPRC, NLL и калибровка, физическая energy MAE на отдельной assay evaluation подвыборке; источник-специфическая coverage, ложное физическое имя, exact fallback.
3. MCAR20/40/50, OFF-only drift, common matched ON+OFF background, ON-A inversion, source leakage, gauge-equivalent world.
4. Expert-intervention active-vs-random при одинаковых бюджетах, не используя TEST Y/oracle при отборе; paired bootstrap и CI cluster-aware по приборам.
5. В отчёте отдельно raw risk benefit, semantic accuracy, explanation fidelity и costs. **Не утверждать победу по AUPRC от одного успешного CI runtime.**

## Связанные peer-reviewed исследования

- Koh et al. (2020), Concept Bottleneck Models, ICML: https://proceedings.mlr.press/v119/koh20a.html
- Espinosa Zarlenga et al. (2022), Concept Embedding Models, NeurIPS: https://papers.nips.cc/paper_files/paper/2022/hash/867c06823281e506e8059f5c13a57f75-Abstract-Conference.html
- Yuksekgonul, Wang, Zou (2023), Post-hoc Concept Bottleneck Models, ICLR: https://openreview.net/pdf?id=nA5AZ8CEyow
- Gresele et al. (2020), Incomplete Rosetta Stone, UAI: https://proceedings.mlr.press/v115/gresele20a.html
- Pandeva and Forré (2023), Multi-View ICA with Shared and Individual Sources, UAI: https://proceedings.mlr.press/v216/pandeva23a.html

**Публикационное ограничение:** пока все физические датчики/эталоны искусственные; не говорить о клинической проверке, новой универсальной теореме или конкурентном SOTA. Исследовательская позиция — воспроизводимые негативные и позитивные границы physical concept semantics.
