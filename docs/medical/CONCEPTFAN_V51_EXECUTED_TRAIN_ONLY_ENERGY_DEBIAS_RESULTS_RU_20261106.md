# ConceptFAN V51 — выполнена независимая bias-калибровка энергии по физическому assay

**6 ноября 2026. Выполнено и воспроизведено.** [Draft PR #48](https://github.com/fims9000/FuzzyAttentionNetworks/pull/48), [frozen protocol до нового TEST](./CONCEPTFAN_V51_FROZEN_TRAIN_ONLY_ENERGY_DEBIAS_PROTOCOL_20261106.md), успешный [GitHub Actions run 42265205302](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/42265205302) на SHA \`e435a9f995413814981bd1f9577e5caaadf02b72\`, 15 regression/algebraic tests. Артефакт \`conceptfan-v51-train-only-energy-calibration\` содержит неизменные sensor/physical assay TEST receipts, параметры, VALID решения и все отрицательные результаты. **Это только новые синтетические приборы, не реальные пациенты, и не улучшение native Transformer AUPRC.**

## Цель, честная гипотеза

В V50 с 5% sparse TRAIN reference и 160 дополнительными VALID физическими assay реконструкция физического concept E1 дала Pearson ~.864, но energy MAE ~.379 при MCAR40. Известная регрессионная причина возможной систематической ошибки: \(\operatorname{Var}(\mathbb E[S_1\mid X])\) в общем случае меньше \(\operatorname{Var}(S_1)\). V51 проверяет *по полностью новым TEST*, может ли калибровка траекторной энергии \(\hat E=\operatorname{Var}_{t,ddof1}\hat s_1\) по **те же TRAIN** noisy external source assays снизить физическую MAE без дополнительной TEST аннотации.

Протокол зафиксировал три варианта: старый \`V50_RAW\`, положительную через-ноль масштабную \`V51_SCALE\`, ограниченную линейную \`V51_AFFINE\`. Внешний независимый assay имеет паспортную ошибку noise variance \(0.16^2=0.0256\), поэтому целевое TRAIN значение \(\max\{0,\operatorname{Var}_t(a)-0.0256\}\). **Это noisy physically measured reference, не скрытое идеальное E.** Все коэффициенты \(\alpha,\beta\) формируются TRAIN-only; условие допуска поправки по двум раздельным VALID80: MAE внешнего noisy assay после коррекции ≤ raw+0.015 в каждом VALID. Иначе эффективная энергия **точно старая V50 raw**, а не другая post-TEST отрегулированная функция.

## Исполнено и независимо пересчитано

- Совершенно новые устройства **20263307/09/21/23**, каждое TRAIN1200 + VALID400+400 + untouched TEST600; для каждого два наблюдательно эквивалентных физических gauge мира с одинаковыми X/Y, но разным именованным источником 1.
- 3 оплаченных бюджета physical source assays: **60, 240, 1200 TRAIN эпизодов + 80 VALID1 + 80 VALID2** для каждого gauge; 24 независимых source calibrations и energy-correction receipts. Никакой скрытый latent H/S/E не использован при fit или gate.
- **216** TEST source-energy model rows (4 устройства × 2 gauge × 3 assay budgets × 3 readout regimes × 3 variants), **72** заранее зарегистрированных paired 95% bootstrap CI по 399 ресэмплов на одинаковых TEST600, без коррекции множественных сравнений. Три stress режима: healthy MCAR20, healthy MCAR40 и unseen amplifier×1.5.
- **15** tests прошли. Независимый аудит заново обучил все **24** TRAIН-only source Ridge + energy maps, проверил обе VALID gate и точно пересчитал **648** TEST оценок MAE/RMSE/Pearson + 72 paired CI: **максимальная абсолютная ошибка 0.0**, включая **39** точных возвращений к прежней V50 energy после VALID отказа.
- Предсказательная native K2/Transformer/HGB голова, вероятности задач и подписанные риск-вклады **вообще не модифицировались**: это улучшение правильности физической *энергии* ранее откалиброванного источника, не доказательство увеличения AUPRC.

## Основной результат: TEST healthy MCAR40, mean MAE (меньше лучше)

| Дорогие TRAIN physical assays | Исходная V50 raw | V51 positive SCALE | V51 bounded AFFINE | Лучший относительный выигрыш MAE |
|---|---:|---:|---:|---:|
| 60 (5%) +160 VAL | .40072 | **.36119** | .37083 | **9.9%** (SCALE) |
| 240 (20%) +160 VAL | .39065 | .36288 | **.34354** | **12.1%** (AFFINE) |
| 1200 (100%) +160 VAL | .39121 | .35460 | **.34807** | **11.0%** (AFFINE) |

Это арифметические средние восьми **связанных** gauge-reconstructions четырёх реальных синтетических устройств (не восемь независимых внешних приборов). Истинное latent source E использовалось **только в итоговом аудите**. В прогоне оба V51 исправления были TRAIN+dual-VALID допустимы одновременно в 15/24 device-gauge-budget случаях; SCALE допустим в 19/24, AFFINE в 16/24. При недопуске поправка оставляла прежний результат V50. У SCALE и AFFINE разные зоны применимости — нельзя заявить, что одна из них универсально лучше.

Среди 8 условных per-device-gauge bootstrap интервалов при budget=5% сравнение \(\Delta\mathrm{MAE}(\mathrm{SCALE}-\mathrm{RAW})\) исключает ноль у 6, а \(\Delta\mathrm{MAE}(\mathrm{AFFINE}-\mathrm{RAW})\) — у 4. При budget=20% AFFINE−RAW исключает ноль у 6/8, при budget=100% у 6/8. Эти TEST CIs **не исправлены за множественные гипотезы**, два gauge внутри одного прибора зависимы, не следует интерпретировать 6/8 как шесть независимых подтверждений.

Pearson corr source-energy vs истинный E при MCAR40 не меняется после принятой положительной аффинной/scale коррекции — при budget5% среднее **.88669**, при budget20% **.88873**, при budget100% **.88947**. Улучшилась точность численного уровня energy, а не обнаружение нового физического направления. Сравнение с V50 (.864 Pearson) по *другим случайным приборам* не является парным приростом и не подтверждает new architecture improvement.

## Отрицательные контроли, границы применимости

- **TRAIN/VALID assay отказ:** за три TEST режима 39 индивидуальных model-row прогнозов гарантированно совпали побитово с raw V50: несовместимое измерение не подменяло энергию.
- **1.5× amplifier sensor shift:** на 4 новых приборах в обоих gauge и для всех budget внешняя scoped-калибровка **отозвана (24/24 калибровок)**, не называла источник физически валидным. На healthy MCAR20 и MCAR40 отказов по readout shift **0/24**. Физический идентификатор устройства и reference остаются доверяемой **внешней аттестацией**, не проверенным нашим кодом фактом.
- **Точный rotational two-world counterexample V50 сохранён:** два мира имеют одинаковые измерения и бинарные исходы, но разный физически именованный концепт. Ни меньшее MAE, ни a priori blur/diffusion Transformer не снимают эту неоднозначность без метролого-структурного условия.
- Если шум внешнего assay **неизвестен** или отклоняется от 0.16, energy debias может внести систематическое смещение; это **не** доказано robust ко всем приборам. Внешние 160 VALID assays, особенно при budget=60 TRAIN, — важная и раскрытая стоимость supervision.

## Научный смысл и корректное продолжение

В серии V44–V49 было показано: masked измерение помогает AUPRC, но физическая идентичность статистического источника опасно ломается при OFF-only drift; human-in-the-loop работает лишь там, где существует независимый метрологический источник. V50–V51 добавили необходимую *semantic provenance contract* и ограниченную калибровку физического уровня concept. Это реальный инженерный компонент для explainable ConceptFAN, но не полный гибкий self-discovering Transformer и не клинически валидированная система.

Следующая верифицируемая интеграция: ***соединить именно условно откалиброванный, provenance-checked physical concept с native signed-only decision head***, но не использовать внешнее ASSAY на TEST, а risk-training и validation проводить по новым TEST меткам. Сравнить с V47 frozen/joint, strong raw HGB на одинаковом Y budget и с равным доступом к external physical assay, обязательно сохранять unknown OFF drift, perfect gauge-twins, missingness and semantically abstained cases. До такого эксперимента **не следует утверждать, что V51 улучшает конечный риск или SOTA**.

**Draft PR #48 — не объединять с master без физической независимой проверки и code review.**
