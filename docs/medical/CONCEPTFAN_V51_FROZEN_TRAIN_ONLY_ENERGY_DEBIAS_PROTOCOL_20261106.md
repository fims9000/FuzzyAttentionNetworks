# ConceptFAN V51 — debias наблюдаемой энергии концепта без доступа к TEST physical truth

**Предрегистрация 6 ноября 2026, ДО нового V51 TEST**. V50 на независимых четырёх синтетических приборах [подтвердила](../medical/CONCEPTFAN_V50_EXECUTED_IDENTIFIABILITY_RESULTS_RU_20261106.md) точную observational rotation ambiguity и показала условную anchor calibration с 60 TRAIN physical assay и 160 VALID assays: Pearson с latent E1 .864, energy MAE .379 на MCAR40. Новая гипотеза: у обыкновенного Ridge траекторная variance обычно занижена из-за условного усреднения; **исключительно TRAIN/VALID метролого-статистическая калибровка самой энергии** должна снизить ошибку без доступа к TEST E1, без изменения native риск-модели и без фиктивного физического сертификата. В процессе разработки V51 разрешено использовать только V50 результаты для формирования параметров; V51 TEST фиксируется отдельно и не участвует в подборе.

## Вход и неизменная идентификационная граница

Взять исходную V50 архитектуру \`independent_anchor_certificate_v50\`: два независимых 2-канальных вида реальных показаний 16×4, Boolean masks, Ridge alpha10 на sparse отдельном физически аттестованном source-1 TRAIN assay \(a=s_1+\eta\). В том же synthetic генераторе независимый физический assay имеет **ИЗВЕСТНУЮ по паспорту** дисперсию ошибки \(\operatorname{Var}(\eta)=0.16^2\), а не узнанную по TEST latent. Предсказанная траектория \(\hat s_1\) даёт исходную \(\hat E_{\mathrm{raw}}=\operatorname{Var}_{t,ddof1}\hat s_1\), чем и ограничена V50.

Только на тех TRAIN эпизодах, где уже был куплен независимый source assay (те же 5%,20%,100% TRAIN), вычислить аудируемый noisy external reference
\[
E^\mathrm{assay}=\max\{0,\operatorname{Var}_{t,ddof1}(a_t)-0.0256\}.
\]
Это **оценка шумного внешнего измерения**, а НЕ идеально истинный скрытый латент E. Никаких скрытых S/H/E не использовать при fit. Два заранее фиксированных корректировщика, плюс исходный V50:
1. **V50_RAW:** \(\hat E_\mathrm{raw}\), без поправки;
2. **V51_SCALE:** \(\hat E_\mathrm{scale} = \alpha \hat E_\mathrm{raw}\), \(\alpha = \operatorname{clip}_{[0.5,2.0]}\sum_i\hat E_{\mathrm{raw},i}E^\mathrm{assay}_i / (\sum_i\hat E_{\mathrm{raw},i}^2+10^{-8})\), \(\alpha\ge0\);
3. **V51_AFFINE:** \(\hat E_\mathrm{affine}=\max(0,\alpha\hat E_\mathrm{raw}+\beta)\), где \(\alpha,\beta\) получены \`Ridge(alpha=5.0,fit_intercept=True)\` **только на тех же TRAIN anchor эпизодах**, затем \(\alpha\) ограничен [0.5,2.0], \(\beta\) [−0.30,0.30].

Оба корректировщика НЕ требуют нового измерения TEST, не меняют прежний Transformer, HGB, frozen K2, q или логику прогнозирования. В каждой из двух непересекающихся VALID по 80 дополнительных дорогих assay эпизодов оценить отдельно MAE энергетического предсказания **против noisy independent \(E^\mathrm{assay}\)**, а не против скрытого true E; допуск поправки только при MAE_valid_adjusted ≤ MAE_valid_raw + 0.015 **на обеих VALID** и при исходной V50 аттестации TRAIN+VALID trajectory corr≥.75. При недопуске использовать точный V50 raw energy; вместе со значением сохранять \`CORRECTION_NOT_VALIDATED\`. При смене физического device/reference ID или сильном readout covariance shift давать только статистический proxy / revocation, независимо от качества MAE.

## Синтетические независимые данные и TEST

Четыре новых прибора **20263307/09/21/23**, каждый 1200 TRAIN/400 VALID1/400 VALID2/600 untouched TEST, 2 rotationally equiv physical gauge worlds, input 16×4, true instrument assay noise .16, 3 уже зафиксированных TRAIN reference budgets 60/240/1200 эпизодов +160 VALID assay, MCAR20 original/MCAR40 original, TEST-only amplifier x1.5 (чтобы проверить отзыв scope). Smoke: device20263307 TRAIN500/VALID200+200/TEST240. Все observation X,Y/masks идентичны у двух gauge worlds, физические target source1 energy различны; без внешнего источника физический label всё ещё неидентифицируем. Верхняя граница на именование и exact V50 risk-label refusal сохранена.

Для каждой из 4 devices × 2 gauge × 3 budgets заранее зарегистрировать 3 парных TEST bootstrap 95% CI (399 patient resamples):
- MAE(V51_SCALE)−MAE(V50_RAW) при healthy MCAR40;
- MAE(V51_AFFINE)−MAE(V50_RAW) при healthy MCAR40;
- MAE(V51_AFFINE)−MAE(V51_SCALE) при healthy MCAR40.
Итого **72 CI**; smoke 199 resamples. Заранее смотреть также знак/качество корректировок VALID, истинные TEST hidden source E (AUDIT ONLY), MAE/RMSE/corr относительно этого источника во всех трёх TEST режимах, число VALID отказов, экономию assay, scope false alarm и known amplifier shift. **Нельзя использовать TEST E для настройки шкалы, gamma или для решения об актуальности концепта.** Сохранять все TEST-patient energy arrays, цены reference, TRAIN-only coef/VALID gates, все четыре прибора и все отрицательные результаты. Отдельный независимый audit script должен перепроверить регрессию и 72 CI, не импортируя генератор или методы из runtime.

**Границы:** положительное снижение физической MAE с идеальным внешним source-specific прибором не означает автоматического discovery и не доказывает каузальность, nonlinear robustness, физический identity в новых датчиках. Адаптация energy на TRAIN не заменяет физический assay на реальной установке. Возможность тестовой смены gauge без изменения X останется фундаментально неустранимой только статистикой. В предыдущей V50 AUPRC HGB .896 не удостоверяла никакого имени concept: контроль обязателен, но V51 не изменяет риск, поэтому не следует заявлять улучшение AUPRC.
