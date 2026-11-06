# ConceptFAN V46 — условный контроль ON/OFF согласованности без нового физического датчика

**6 ноября 2026. Зарегистрировано до просмотра новых V46 TEST.** Родительский V45 показал: OFF-only контроль на сильном OFF-only дрейфе обнаруживает 99.54% эпизодов, но при одинаковой ON/OFF помехе U8 отклоняет 100% и разрушает полезный прогноз. V46 проверяет новую измерительную гипотезу **НА НОВЫХ ПРИБОРАХ**, не перенастраивает результат V45 по его TEST, не меняет PIERE.

## Архитектура и математическое правило

Новый источник по-прежнему определяется без outcome Y через частичные межвидовые ковариации V44 с TRAIN-only идентификацией направлений, калибровкой пар и выбором реально измеренной пары. Риск: прежняя настоящая native K2 Temporal ConceptFAN + обученный на TRAIN линейный концептный вклад в том же torch.forward, при отказе точно K2.

Два разных измерительных свидетельства:
1. **OFF variability**: независимая от Y временная дисперсия фактически наблюдаемых OFF каналов 4:10 вокруг их эпизодного среднего, минимум 4 реальных временных точки по каждому из шести, усреднение по шести.
2. **ON−OFF mismatch variability**: для каждого канала 4:10 берём только одновременно наблюдённые ON и OFF; рассчитываем временную дисперсию разности, минимум 4 реальных парных момента для канала, минимум 4 из 6 каналов, усредняем исключительно поддержанные каналы. Никакой подстановки выдуманных значений.

Healthy TRAIN1000 отдельно задаёт два порога: OFF score **99-й перцентиль**, mismatch score **95-й перцентиль**. Новая **совместная тревога только при обоих аномальных показателях**:
\[
A_i = \mathbb I[R^{OFF}_i>Q_{.99}^{TRAIN}(R^{OFF}) \;\land\;
R^{ON-OFF}_i>Q_{.95}^{TRAIN}(R^{ON-OFF})].
\]
Недостаточный реально измеренный OFF или ON−OFF => reject дополнительного концепта и точно оригинальный K2 (fail closed). На двух независимых healthy VALID400 требуется >=90% monitor support и <=5% ложных тревог среди поддержанных. Пороги, формула и правила **НЕ** выбираются по V46 TEST; исходы Y, генераторские H/S/U/E, клинические pseudo-имена и внешние V43 nuisance монитор никогда не входят в диагностику. Неподдержанный источник и несогласие независимых пар V44 всё равно приводят к K2.

## Новый эксперимент

Устройства **20262107, 20262109** (linear AR), **20262121, 20262123** (nonlinear switching). Это новые случайные приборы и TEST-эпизоды того же исходного генераторного семейства, не внешние клинические данные. TRAIN1000, VALID1=400, VALID2=400, TEST=600 на прибор; 3 независимые native K2 fits seeds11/23/37, 16×12 ON/OFF. Smoke device 20262107 TRAIN360/VALID150+150/TEST240.

Методы:
- K2;
- V44_PARTIAL_VIEW (без контроля);
- V45_OFF_GUARD (предыдущий, OFF-only);
- **V46_PAIRED_CONCORDANCE** (новое совместное правило);
- RAW_DIFF_HGB_TRAIN1000;
- RAW_DIFF_HGB_ALL1800;
- RAW_DIFF_HGB_ALL1800_PAIRED_GUARD (HGB1800 + *тот же V46 контроллер* + точно K2 на отказе; это гибрид с дополнительной моделью, а не чистый HGB).

Сценарии: healthy_energy, matched common U8 ON+OFF, unknown OFF-only U4 drift, unknown ON-A polarity inversion, OFF source leak 50%, complete original K2 outcome dictionary, known VALID2 OFF drift. Пропуски: MCAR20, MCAR40, MCAR50, missing_view_A, missing_reference. Одинаковая frozen K2 TEST маска на всех вариантах. Не брать Y/R на TEST для fitting и decisions.

До TEST фиксируем 6 paired AUPRC bootstrap 95% CI (399 bootstrap, smoke 199) на каждом приборе: matched-common/MCAR40 V46−V45; unknown-OFF/MCAR40 V46−V45; unknown-OFF/MCAR40 V46−K2; healthy/MCAR40 V46−V44; healthy/MCAR40 V46−HGB1800; unknown-OFF/MCAR40 V46−HGB1800_PAIRED_GUARD. Отдельно TEST AUPRC, NLL, observed concept usage, gate false alarm, missed OFF drift, overrejection matched common, negative ON inversion/OFF leak, real patient-level measured monitor receipts, exact K2 fallback, comparison on all four приборов. Не отбирать удачные TEST; сохранить все ошибки, файлы вероятностей, приборы, CI, TRAIN/VALID сертификаты, native weights и SHA.

**Ограничения:** это обычный двухиндикаторный unsupervised change detector и классический ON/OFF subtraction, не новая идентификационная теорема и не доказательство физической или клинической истинности концепта. При nonlinear pre-readout g(x) matched ON/OFF может не отменяться, а источник может протечь через OFF; ни fuzzy order T3, ни internal agreement не гарантируют истинность. Нельзя заявлять превосходство над HGB до независимых TEST и полного аудита. **Draft, без merge master.**
