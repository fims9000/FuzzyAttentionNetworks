# ConceptFAN V48 — ограниченная log-odds связь измеренного fuzzy-концепта с риском

**Предварительная регистрация 6 ноября 2026, ДО просмотра новых V48 TEST.** V47 реально реализовала совместный measured-proxy anchored K2 Transformer+концептный head. При этом линейная связь \`gamma*(q3-.5)\` насыщает вклад, поскольку q3 есть sigmoid от калиброванной измеренной энергии. V47_JOINT_PROXY лишь описательно выиграла у V44 0.00149 AUPRC на 4 старых приборах, а masked-consistency ablation не была полезна. Эта гипотеза формулируется отдельно для **новых** субъектов и приборов, нельзя подбирать пороги на увиденных V47 TEST.

## Frozen function and architecture

Для нового статистического измеренного fuzzy-концепта \`q3 in [0,1]\` используем **две явно зарегистрированные головы с одинаковым набором параметров**:
- V47_JOINT_LINEAR: \`e(q)=q-0.5\` (буквально исходный V47, lambda_consistency=0).
- V48_JOINT_LOGODDS: \`e(q)=clip(logit(clip(q, 0.001, 0.999)), -4, 4)/8\`. Это центрированное неубывающее ограниченное evidence в [−0.5,0.5], не вероятность и не новый theorem fuzzy type reduction. Отдельная нормализация нужна, чтобы запретить непредсказуемо большие рисковые вклады в хвостах; \`q=0.5\` -> \`e=0\`, \`q\` сохраняет строгую монотонность в незажатой области.
- Сохранение полученного q и e для независимого аудита. Преобразование реализуется без Y/TEST и не меняет исходный V44 detector; только значение, поступающее в **строгую** подписанную концептную голову.

\[
z=b+w_1(c_1-.5)+w_2(c_2-.5)+\gamma e(q_3)g_3
\]

Transformer видит **только четыре** измеренных известных сенсора и их Boolean mask. Новый q3 только из 6 неизвестных каналов ON−OFF по зафиксированным outcome-blind V37/V44 направлениям/калибровке; ни hidden H/S/U/E, ни прямой raw shortcut. При \`g3=0\`, \`VALID gate rejected\` или OFF missing точно original frozen K2. Физический смысл q3 не сертифицирован.

Оба joint варианта запускают одинаковый исходный pretrained K2 и TRAIN gamma-V44; gamma-init для V47 линейного варианта = исходная TRAIN gamma, для нового V48 logodds = **2× TRAIN gamma**, потому что центральная производная \`e(logit(q))\` по \`logit(q)\` составляет 1/8, а \`d(q-.5)/d(logit(q))=1/4\`. Протокол не подразумевает параметрического преимущества; у обеих голов те же 3 линейных веса + bias. Обе модели совместно обучают Transformer и подписанный риск 8 эпох AdamW lr=.001, weight_decay=.001, batch128, \`L=BCE+5×observed known-proxy MSE\`. **Дополнительного consistency веса нет** в обоих вариантах, чтобы сравнить исключительно связь q3 и риска. После TRAIN каждую модель независимо проверять на VALID1/2 по тем же заранее зафиксированным V47 критериям source support≥25%, ΔNLL≥−0.002, ΔAP≥−0.02 и proxy-MSE ratio≤1.25 в каждой когорте. Ни одного TEST outcome для выбора.

## Fresh devices and fixed TEST cases

Четыре совершенно новых synthetic приборных генератора: **20262707,20262709** (linear AR), **20262721,20262723** (nonlinear switching), каждая TRAIN1000, VALID1=400, VALID2=400, TEST600; 3 независимые seed 11/23/37, 16 временных шагов/12 датчиков. Smoke device20262707 TRAIN360 VALID150+150 TEST240 seed11.

6 методов: original K2; V44_FROZEN (1000 Y+dual VALID); V47_JOINT_LINEAR; V48_JOINT_LOGODDS; RAW_DIFF_HGB_TRAIN1000; RAW_DIFF_HGB_ALL1800 (uses 1800 Y including VALID, larger outcome label budget). 4 scenarios: healthy_energy, matched common_U8, unseen OFF-only drift, OFF_source_leak 50%; 4 missingness patterns: MCAR20/40/50 и completely missing OFF. TEST known sensor mask exactly same for all methods per device/case; same subjects and labels per device. No causal source identity claim.

**Primary preregistered 5 paired TEST patient bootstrap AUPRC CIs on each of 4 devices, 399 repetitions (smoke 199):**
1. healthy/MCAR40 V48_LOGODDS − V47_LINEAR;
2. healthy/MCAR40 V48_LOGODDS − V44_FROZEN;
3. healthy/MCAR40 V48_LOGODDS − HGB1000;
4. healthy/MCAR40 V48_LOGODDS − HGB1800;
5. unknown_OFF_drift/MCAR40 V48_LOGODDS − original K2.

Evaluate AUPRC/NLL/Brier, K2 exact fallback, source use, VALID approvals, measured known-proxy MSE/seed stability, source q/e distribution, all four devices and failures. Full frozen patient-level predictions, risk e, original q, outcome Y, trained weights, >600 source/use masks, independent audit. Correction for multiplicity *not supplied*; do not infer population-wide superiority from individual intervals.

Potential **predeclared limitation**: stretched logodds can raise quality when source is genuine, but also **amplify an unrecognized OFF-only drift**; semantic provenance and instrument reliability remain independent concerns. Do not retune \`clip\` or gamma after TEST V48, never select only successful devices, leave Draft until independently reproduced and reported.
