# Plan — entraînement long du modèle RL d’émissions avec demande variée

Statut : **étapes A et B réalisées** (demandes générées, code et tests).
Les étapes C, D et E ne sont pas lancées. Aucun modèle existant n’est modifié.

Réalisé :

- `flowmind/demands.py` : chargement des lots vérifié par `index.json`
  (rôle et SHA-256), séparation des lots, paires de validation.
- `scripts/generate_demands.py` : génération et contrôle FixedTime,
  sortie dans `scenarios/3x3grid/demand/`.
- `scripts/train_emissions.py` : options `--train-demands`,
  `--validation-demands`, `--demand-seed`, `--n-steps`, `--batch-size`,
  `--n-epochs`, `--learning-rate`, `--linear-lr`, `--gamma`, `--gae-lambda`,
  `--ent-coef`. Sans ces options, le comportement précédent est inchangé.
- `tests/test_demands.py` : 10 tests.

## 1. Pourquoi les entraînements précédents ne progressaient pas

Constat vérifié dans le code et les données :

- `scenarios/3x3grid/routes14000.rou.xml` contient une liste fixe de 7 000
  véhicules, un départ toutes les 2 s, avec des itinéraires figés.
- `SumoCorridorEnv` démarre toujours au premier départ (`_get_first_depart`),
  donc chaque épisode de 600 s rejoue **les mêmes ~300 véhicules**.
- `--seed` ne change que l’aléa interne de SUMO (conduite), pas la demande.
  Les seeds d’entraînement, de validation et de test voient donc le même trafic.
- Budget très faible : 2 048 à 15 488 pas (12 à 64 épisodes), avec
  `n_steps=64`, `batch_size=32`, `n_epochs=4`.
- Les gains mesurés (0,02 à 0,15 %) sont plus petits que l’écart de
  reproductibilité observé sur une seed de test (environ 0,3 % de CO₂).

Conséquence : le modèle apprend un seul scénario et ne peut pas généraliser.
Ajouter des pas sans varier la demande a déjà dégradé V4 et V6.

## 2. Objectif de la campagne

Entraîner une nouvelle politique `duration` sur **≥ 100 000 pas** (environ
600 épisodes) avec des demandes différentes à chaque épisode. Ensuite, la
comparer aux feux classiques sur des demandes jamais vues.

Seuil de succès fixé **avant** l’entraînement :

- Médiane appariée des quatre émissions sous 1, avec des baisses
  supérieures à 1 % (au-dessus du bruit observé).
- Aucune baisse des arrivées, ni collision, ni téléportation.
- Résultat stable sur au moins 8 demandes de test sur 10.

Sinon, le modèle reste non déployé, comme V6 et V7.

## 3. Étape A — Générer les demandes (nouveau script, aucun entraînement)

Nouveau script `scripts/generate_demands.py`, qui appelle
`randomTrips.py` de SUMO (`$SUMO_HOME/tools`) sur `3x3Grid2lanes.net.xml` :

| Paramètre | Valeurs | Effet |
|---|---|---|
| `--period` | 1,2 · 1,6 · 2,0 · 2,5 · 3,0 s | Charge forte à faible |
| `--fringe-factor` | 1 · 5 · 10 | Trafic local ou traversant |
| `--seed` | unique par fichier | Origines et destinations différentes |
| `--end` | 600 (+ 60 s de marge) | Couvre un épisode |
| Profil optionnel | `--period` à plusieurs valeurs | Montée et descente d’une pointe |

Lots strictement séparés (seeds `randomTrips` différentes) :

- `scenarios/3x3grid/demand/train/` : 40 fichiers.
- `scenarios/3x3grid/demand/validation/` : 6 fichiers.
- `scenarios/3x3grid/demand/test/` : 10 fichiers, réservés au test final.

Contrôle : chaque fichier est simulé une fois avec FixedTime. Les demandes
qui produisent un blocage (téléportations) ou presque aucun trafic sont
écartées. Les empreintes SHA-256 sont listées dans `demand/index.json`.

## 4. Étape B — Modifications du code d’entraînement

Dans `scripts/train_emissions.py` (les anciens modèles restent compatibles) :

1. **`--train-demands` / `--validation-demands`** : dossiers de routes.
   `EmissionTrainingEnv.reset()` tire un fichier différent à chaque épisode
   (aléa contrôlé par une seed enregistrée), puis le passe à
   `SumoCorridorEnv(route_file=...)`. La seed SUMO varie aussi.
2. **Calibration** des `scales_mg_s` : moyenne FixedTime sur un échantillon
   de 10 demandes d’entraînement, puis figée dans le manifeste.
3. **Référence de validation** : FixedTime calculé par couple
   (demande, seed), comparaison appariée comme aujourd’hui.
4. **Hyperparamètres PPO exposés en arguments** (aujourd’hui codés en dur) :

   | Paramètre | Actuel | Proposé |
   |---|---|---|
   | `n_steps` | 64 | 1 024 (environ 6 épisodes) |
   | `batch_size` | 32 | 256 |
   | `n_epochs` | 4 | 10 |
   | `learning_rate` | 3e-4 ou 1e-4 fixe | 3e-4 avec décroissance linéaire |
   | `gamma` | 0,99 ou 1 | 0,99 |
   | `gae_lambda` | 0,95 | 0,95 |
   | `ent_coef` | 0 | 0,01 (garde l’exploration) |

5. **Manifeste** : liste et empreintes des demandes, hyperparamètres,
   seed de tirage, qualification `experimental_long`.
6. **Tests unitaires** : tirage reproductible des demandes, séparation
   train/validation/test, refus d’une demande de test pendant l’entraînement.

Inchangés : observation, SafetyGate (vert 10 à 35 s, rouge max 30 s),
protocole SUMO (`ballistic`, un thread, `--random false`), sélection du
checkpoint (collisions, puis téléportations, puis pire ratio).

Choix volontaires pour isoler l’effet de la demande variée :

- **Réseau neuf**, pas de reprise de V5 : la politique V5 est restée
  proche de 30 s et n’a vu qu’un seul trafic.
- **Récompense `worst_interval`** (quatre émissions à poids égal), mais avec
  `gamma=0,99` au lieu de 1 pour réduire la variance sur 170 décisions.
- Les files d’attente **ne sont pas ajoutées** à l’observation dans cette
  campagne. Ce sera une campagne suivante si la demande variée ne suffit pas.

## 5. Étape C — Essai court de vérification

```powershell
python scripts/train_emissions.py --output runs/models/emission_long_smoke --seconds 600 --timesteps 4096 --validation-every 2048 --train-demands scenarios/3x3grid/demand/train --validation-demands scenarios/3x3grid/demand/validation --action-mode duration --reward-mode worst_interval --n-steps 1024 --batch-size 256 --n-epochs 10 --ent-coef 0.01 --gamma 0.99 --linear-lr --learning-rate 3e-4
```

Sans `--learning-rate`, `worst_interval` garde 1e-4 comme avant.

### Résultat de l’essai court (27 octobre 2026)

Le modèle est dans `runs/models/emission_long_smoke/` et le journal dans
`runs/train_long_smoke.log`.

- **Le parcours fonctionne.** 4 096 pas, 28 épisodes sur 25 demandes
  différentes (toutes les charges), 6 références de validation, aucune
  collision ni téléportation. Environ 138 décisions par épisode.
- **Vitesse.** Environ 0,21 s par décision, validations comprises : 100 000
  pas prendraient environ **7 h** (et non 13 h).
- **La politique n’a presque pas bougé.** La probabilité de choisir 30 s
  reste à 0,41 sur les 9 carrefours, soit sa valeur initiale. En validation, le
  modèle choisit toujours l’action la plus probable, donc toujours 30 s, comme
  les feux classiques. Les ratios restent entre 0,9996 et 1,0000.
- **La récompense dépend surtout de la charge.** Moyenne par épisode : −400
  (3 s entre véhicules) à −1 323 (1,2 s), et −1 066 à −1 585 au sein de la
  charge la plus forte. La pénalité de dépassement au carré est presque
  toujours active pour les fortes charges et jamais pour les faibles. L’effet
  d’un choix de feu (moins de 1 %) est noyé dans ces écarts.

Corrections recommandées avant l’étape D :

1. **Récompense normalisée par demande** : un débit de référence FixedTime
   pour chacune des 40 demandes d’entraînement, déjà simulées une fois. Un
   ratio de 1 signifie alors « aussi bien que les feux classiques sur ce
   trafic ». Les observations gardent les références globales.
2. **Initialisation neutre pour cette campagne** : retirer ou réduire le
   biais initial vers 30 s. Sinon, le choix le plus probable reste 30 s
   longtemps, et la validation ne voit aucun changement.
3. Refaire l’essai court après ces corrections avant de lancer 100 000 pas.

### Essai court n° 2 : récompense par demande et départ neutre (28 octobre 2026)

Options ajoutées : `--per-demand-reward` (référence FixedTime pour chacune des
40 demandes d’entraînement, les observations gardent les références globales)
et `--neutral-init` (départ uniforme sur 20/25/30/35 s). Le modèle est dans
`runs/models/emission_long_smoke2/` et le journal dans
`runs/train_long_smoke2.log`.

Moyenne des 6 demandes de validation, par rapport aux feux classiques :

| Validation | CO₂ | NOx | Carburant | PM | Arrivées |
|---|---:|---:|---:|---:|---:|
| Pas 0 (départ neutre, en pratique 20 s partout) | −7,1 % | −7,8 % | −7,1 % | +1,8 % | +2,8 % |
| Pas 2 048 (retenu) | −0,04 % | +0,06 % | −0,04 % | +1,1 % | +0,8 % |
| Pas 4 096 | −1,9 % | −1,7 % | −1,9 % | +1,1 % | +1,1 % |

- **La normalisation par demande fonctionne.** La récompense moyenne
  par charge va maintenant de −652 à −713, contre −400 à −1 323 auparavant.
- **La politique n’a pas encore appris.** Les probabilités restent à 0,25
  pour chaque durée. Le choix déterministe dépend de différences minimes, donc
  les résultats de validation changent fortement d’un checkpoint à l’autre.
  La récompense moyenne ne progresse pas (−659,5 sur la première moitié des
  épisodes, −663,9 sur la seconde).
- **Les PM augmentent dans presque tous les cas.** Au pas 0, les 6 demandes
  ont −4,5 % à −10,3 % de CO₂ et +0,5 % à +2,8 % de PM. Au pas 4 096, 5 demandes
  sur 6 ont des PM en hausse. Raccourcir le vert réduit le CO₂ et les NOx, mais
  semble augmenter les PM : c’est un compromis physique, pas seulement un
  problème d’entraînement.

Conclusion : la politique apprise ne garde pas la baisse de CO₂ et de NOx sans
augmenter les PM. Avant 7 h d’entraînement, il est recommandé de mesurer des
feux fixes à 20, 25 et 35 s sur les 6 demandes de validation. Cela dira si une
durée, ou un mélange de durées, peut réduire les quatre émissions à la fois.

### Mesure des feux fixes de 15 à 40 s (28 octobre 2026)

36 simulations de 600 s : 6 durées × 6 demandes de validation, mêmes seeds que
la validation. Résultats dans `runs/reports/fixed_durations_20261028/`.
Moyenne par rapport au vert fixe de 30 s :

| Vert fixe | CO₂ | NOx | Carburant | PM | 4 baisses | Arrivées |
|---|---:|---:|---:|---:|---:|---:|
| 15 s | −7,7 % | −8,7 % | −7,7 % | +2,3 % | 0/6 | +53 |
| 20 s | −7,1 % | −7,8 % | −7,1 % | +1,7 % | 0/6 | +44 |
| 25 s | −2,1 % | −2,3 % | −2,1 % | +1,5 % | 0/6 | +28 |
| 30 s (référence) | 0 | 0 | 0 | 0 | — | 0 |
| 35 s | +6,5 % | +7,3 % | +6,5 % | +0,4 % | 0/6 | −5 |
| 40 s | +12,2 % | +13,9 % | +12,2 % | −0,7 % | 0/6 | −40 |

Aucune collision ni téléportation. La simulation à 30 s relancée reproduit la
référence de l’entraînement à 0,07 % près (0,3 % pour les PM sur une demande).

Conclusions :

- **Aucune durée fixe ne réduit les quatre émissions, sur aucune demande.**
  Un vert plus court réduit fortement le CO₂, les NOx et le carburant et fait
  arriver plus de véhicules, mais augmente les PM. Seul le vert de 40 s réduit
  les PM, au prix de +12 % de CO₂.
- **Un mélange de durées ne suffit pas non plus.** En combinant 20 s et 40 s,
  il faudrait plus de 63 % de 20 s pour réduire le CO₂, et moins de 29 % pour
  réduire les PM. Les deux conditions sont incompatibles.
- **Les feux fixes à 20 s battent déjà largement tous les modèles RL entraînés**
  sur le CO₂, les NOx et le carburant (−7 % contre au mieux −0,15 % en test),
  avec une hausse de PM comparable à celle des essais RL.

L’objectif « quatre baisses simultanées » n’est donc pas atteignable en
réglant seulement la durée du vert. Deux choix possibles avant l’étape D :

1. **Accepter une tolérance sur les PM** (par exemple +2 % au plus) et viser le
   CO₂, les NOx et le carburant. La référence à battre devient alors le vert
   fixe de 20 s, et pas seulement celui de 30 s.
2. **Garder les quatre baisses** et donner au modèle d’autres actions que la
   durée : choix ou saut de phase, coordination entre carrefours (onde verte)
   pour réduire les arrêts et redémarrages, qui produisent les PM.

## Objectif retenu : tolérance sur les PM (28 octobre 2026)

Décision : viser le CO₂, les NOx et le carburant, accepter au plus **+2 % de
PM** par rapport aux feux classiques (30 s), et **battre le vert fixe de 20 s**.

Ajouts dans `scripts/train_emissions.py` :

| Option | Rôle |
|---|---|
| `--reward-mode pm_tolerant` | Coût par seconde : moyenne des ratios CO₂/NOx/carburant + `pm_penalty` × dépassement des PM au-delà de 1 + tolérance. Ratios par rapport aux feux classiques sur la même demande |
| `--pm-tolerance 0.02` | Hausse de PM tolérée (+2 %) |
| `--pm-penalty 4` | Poids du dépassement des PM |
| `--green-durations 15 20 25 30` | Durées proposées au modèle. 15 s permet d’aller plus loin que la référence à 20 s |
| `--target-green 20` | Seconde référence de validation : feux fixes à 20 s, enregistrée dans `validation_target.json` |

Sélection du checkpoint en mode `pm_tolerant` : d’abord aucune collision ni
téléportation, puis PM dans la tolérance et aucune perte d’arrivées ou de
départs, puis la plus faible moyenne CO₂/NOx/carburant. `validation.csv` ajoute
`target_*_ratio` (par rapport au vert de 20 s) et `beats_target` (CO₂, NOx et
carburant tous sous le niveau du vert de 20 s).

La pénalité des PM est calculée seconde par seconde : c’est une approximation.
Seuls les totaux officiels de la validation disent si la tolérance est respectée.
Avec `--neutral-init`, le choix le plus probable au départ est la première durée
(15 s).

Vérifications : 119 tests passent (9 nouveaux dans `tests/test_pm_tolerant.py`).
Un mini-essai de 128 pas (épisodes de 120 s) a parcouru tout le processus, référence
de 20 s comprise. Ses chiffres ne sont pas significatifs, car les épisodes sont trop courts.

Essai court à lancer avant l’entraînement long :

```powershell
python scripts/train_emissions.py --output runs/models/emission_pm_smoke --seconds 600 --timesteps 4096 --validation-every 2048 --train-demands scenarios/3x3grid/demand/train --validation-demands scenarios/3x3grid/demand/validation --per-demand-reward --workers 6 --neutral-init --action-mode duration --green-durations 15 20 25 30 --reward-mode pm_tolerant --pm-tolerance 0.02 --target-green 20 --n-steps 1024 --batch-size 256 --n-epochs 10 --ent-coef 0.01 --gamma 0.99 --linear-lr --learning-rate 3e-4
```

Entraînement long (étape D) : même commande avec `--output runs/models/emission_pm_v8`,
`--timesteps 120000` et `--validation-every 5120` (environ 7 h).

À vérifier : demandes différentes dans `training_episodes.csv` (nouvelle
colonne `demand`), aucune erreur TraCI, durée réelle par pas.

### Résultat de l’essai court `pm_tolerant` (28 octobre 2026)

Dossier : `runs/models/emission_pm_smoke`, 4 096 pas, aucune collision ni
téléportation. Moyennes sur les 6 demandes de validation :

| Pas | CO₂ / 30 s | NOx / 30 s | PM / 30 s | CO₂ / 20 s | Choix du modèle |
|---|---:|---:|---:|---:|---|
| 0 (départ neutre) | −7,8 % | −8,8 % | **+2,3 %** | −0,7 % | 15 s partout |
| 2 048 | −1,6 % | −1,7 % | +1,7 % | +5,9 % | durées mélangées |
| 4 096 | −1,8 % | −1,8 % | +1,2 % | +5,7 % | 15 s 33 %, 20 s 38 %, 25 s 13 %, 30 s 16 % |
| Vert fixe 20 s (mesure) | −7,1 % | −7,8 % | +1,7 % | 0 | — |

Lecture :

- **L’objectif n’est pas atteint.** Aucun point ne bat le vert de 20 s tout en
  restant sous +2 % de PM. Le départ (15 s partout) bat le 20 s sur le CO₂ de 0,7 %,
  mais ses PM (+2,3 %) dépassent la tolérance : sur 4 demandes sur 6, ils sont au-dessus de +2 %.
- Les points 2 048 et 4 096 respectent la tolérance des PM, mais émettent **5 à 6 % de CO₂
  de plus que le vert fixe de 20 s**. Mélanger les durées d’un carrefour à l’autre
  casse la coordination entre carrefours. Le résultat est moins bon que n’importe
  quel vert court appliqué partout.
- Les probabilités n’ont presque pas bougé (0,25 ± 0,004). Les choix déterministes
  dépendent de très petits écarts : le modèle n’a pas encore appris.
- Le checkpoint retenu (`best.zip`) est le pas 0, car aucun point n’est « faisable ».
  Le pas 0 dépasse la tolérance des PM. Les pas 2 048 et 4 096 ont **1 véhicule parti
  de moins** que la référence (1 900 au lieu de 1 901). Le critère « aucune perte
  de départs » est strict et ne tolère même pas ce bruit d’un véhicule.

Conséquence pour l’étape D : ne pas lancer 120 000 pas avec ce réglage. Pour un
CO₂ comparable, la meilleure solution mesurée reste un vert fixe court et identique
partout. 20 s donne CO₂ −7,1 %, NOx −7,8 % et PM +1,7 %, dans la tolérance. Pistes
avant tout entraînement long :

1. Action coordonnée : une seule durée choisie pour tout le réseau (15/20/25 s),
   au lieu d’une durée par carrefour.
2. Tolérer un écart de ±1 véhicule (bruit) dans le critère des départs.
3. Ou adopter le vert fixe de 20 s comme nouvelle référence classique, sans RL.

## 6. Étape D — Entraînement long

```powershell
python scripts/train_emissions.py --output runs/models/emission_long_v8 --seconds 600 --timesteps 120000 --validation-every 5120 --train-demands scenarios/3x3grid/demand/train --validation-demands scenarios/3x3grid/demand/validation --action-mode duration --reward-mode worst_interval --n-steps 1024 --batch-size 256 --n-epochs 10 --ent-coef 0.01 --gamma 0.99 --linear-lr --learning-rate 3e-4
```

Durée estimée sur ce PC, d’après V7 (environ 0,35 s par décision, un seul
processus) :

- Apprentissage : 120 000 pas, soit environ 11 h 30.
- Validation : 24 passages × 6 demandes × environ 170 décisions, soit environ 2 h.
- **Total : environ 13 h**. Lancer la nuit, dashboard arrêté.

Option pour accélérer (modification supplémentaire) : plusieurs
environnements parallèles (`SubprocVecEnv`, 4 à 8 processus) avec un
`label` TraCI distinct par processus. Durée attendue : environ 3 à 4 h.
Cette option est à valider séparément, car elle touche à la reproductibilité.

Suivi pendant l’entraînement :

- `progress.json` et `validation.csv` à chaque validation.
- La récompense moyenne par tranche de 50 épisodes doit progresser. Si
  elle reste plate après 40 000 pas, arrêter et revoir la récompense ou
  l’observation plutôt que continuer.
- `best.zip` retient le meilleur checkpoint en validation et `last.zip` l’état final.

## 7. Étape E — Test final indépendant

Sur les 10 demandes de test, deux répétitions par méthode, ordre alterné.
Cela donne 40 simulations de 600 s, soit environ 1 h :

```powershell
python scripts/compare_emission_trials.py --model-dir runs/models/emission_long_v8 --demands scenarios/3x3grid/demand/test --seeds 10200 10201 --seconds 600 --repeats 2 --output runs/reports/long_v8
```

(`--demands` est à ajouter à `compare_emission_trials.py`.)

Rapport : médiane et étendue des ratios par demande, arrivées, véhicules
restants, collisions, téléportations, répartition par niveau de charge.
L’activation dans `runs/models/emission_ppo/` n’a lieu que si le seuil de
succès de la section 2 est atteint. L’ancien modèle est alors sauvegardé avant.

## 8. Ordre et estimation globale

| Étape | Travail | Durée machine |
|---|---|---|
| A | Script de demandes + contrôle FixedTime | environ 30 min |
| B | Modifications du code + tests | (développement) |
| C | Essai court | environ 30 min |
| D | Entraînement long | environ 13 h (ou 3 à 4 h en parallèle) |
| E | Test final + rapport | environ 1 h |

### Essai court avec action coordonnée (29 octobre 2026)

Ajouts : `--coordinated` (une seule durée de vert pour les 9 carrefours,
`coordinated_action` dans le manifeste, rejouée de la même façon par
l’évaluation). En mode `pm_tolerant`, un point est accepté s’il manque au plus
1 véhicule parti au total (`departure_gap`). 123 tests passent.

Dossier : `runs/models/emission_coord_smoke`, 4 096 pas, aucune collision ni
téléportation. Moyennes sur les 6 demandes de validation :

| Pas | CO₂ / 30 s | NOx / 30 s | PM / 30 s | CO₂ / 20 s | Acceptable |
|---|---:|---:|---:|---:|---|
| 0 (15 s partout) | −7,9 % | −8,9 % | +2,1 % | −0,7 % | non (PM) |
| 2 048 | −1,1 % | −1,2 % | +1,2 % | +6,6 % | oui |
| 4 096 (retenu) | −3,2 % | −3,6 % | +1,4 % | +4,3 % | oui |
| Vert fixe 20 s | −7,1 % | −7,8 % | +1,7 % | 0 | oui |

Lecture :

- La coordination aide : −3,2 % de CO₂ au lieu de −1,8 % avec une durée par
  carrefour, et le point retenu respecte la tolérance des PM.
- L’objectif n’est toujours pas atteint : +4,3 % de CO₂ par rapport au vert fixe de 20 s.
- Les probabilités ont à peine bougé (0,24 à 0,26). Le modèle choisit 20 s
  partout. En mode `duration`, cela équivaut à un vert réel plus long que le
  vert fixe de 20 s, car la bascule n’a lieu qu’au point de décision suivant.
  À l’inverse, « 15 s » dans ce mode donne à peu près le vert fixe de 20 s.
- Plafond possible : même parfait, un choix de durée commune selon la charge ne
  peut guère faire mieux que le meilleur vert fixe court. Le point 0 en donne une
  idée : environ −1 % de CO₂ par rapport au 20 s, avec des PM à la limite. Un
  entraînement long de 7 h viserait donc au mieux environ 1 % de gain par rapport
  au vert fixe de 20 s.

## Décision : vert fixe de 20 s comme référence classique (29 octobre 2026)

`config.yaml` passe `fixed_time.green` de 30 s à 20 s. Le contrôleur « Sans
agent · feux classiques » du tableau de bord, les évaluations et les futures
références d’entraînement utilisent donc 20 s. Gain mesuré par rapport à l’ancien
vert de 30 s, sur les 6 demandes de validation : CO₂ −7,1 %, NOx −7,8 %,
carburant −7,1 %, PM +1,7 % (dans la tolérance de +2 %), et 44 arrivées de plus.

Conséquences :

- Les rapports et l’historique antérieurs au 29 octobre comparent à 30 s. Les
  évaluations enregistrent le paramètre `fixed_time` utilisé.
- Le modèle actif `runs/models/emission_ppo` garde ses références à 30 s,
  stockées dans son manifeste. Face aux nouveaux feux classiques, il émet environ
  7 % de CO₂ en plus : il n’apporte plus de gain.
- L’entraînement long (étape D) n’est pas lancé. Un futur modèle RL devra battre
  ce vert de 20 s avec d’autres leviers, par exemple des décalages entre carrefours.
