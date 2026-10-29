# Récompense RL et reproductibilité — 22 octobre 2026

Le nouvel entraînement et les comparaisons sont terminés. **L’objectif de
réduire les quatre métriques n’est pas atteint. Le nouveau modèle n’est pas
activé.** La reproductibilité est vérifiée sur plusieurs cas, mais des écarts
résiduels dans SUMO empêchent de la garantir pour toutes les seeds.

## 1. Diagnostic des simulations

Les fichiers de commandes TraCI ont été comparés entre répétitions. Avec
SUMO 1.27.1, des commandes de feux identiques produisaient parfois des vitesses
différentes dès la seconde 79, pour le véhicule 22 dans un virage interne du
carrefour 5. L’écart existait avant le calcul des métriques.

Le verrouillage des threads et un essai isolé avec SUMO 1.28 n’ont pas éliminé
ce comportement. La version installée sur le PC n’a pas été remplacée.

Le protocole courant impose une seed explicite, `--random false`, un thread,
un pas de simulation d’une seconde et l’intégration `ballistic`. Le chemin et
l’empreinte SHA-256 du binaire sont enregistrés dans le protocole. Les résultats
antérieurs au changement d’intégration ne sont pas réutilisés comme références.
La circulation peut différer du protocole précédent : ce changement concerne
à la fois les feux classiques et le RL.

Quatre répétitions classiques de 600 secondes sur la seed 1004 sont identiques
seconde par seconde, décisions comprises. Un audit en processus séparés sur
1005 passe aussi. En revanche, un audit RL sur 10080 présente encore des écarts
malgré des décisions identiques. **Le problème n’est donc pas entièrement
résolu.** La cause interne précise n’est pas démontrée. Un profil avec
`sigma=0` a également été essayé pour le diagnostic ; il n’est pas adopté.

Le nouvel audit du candidat V7 sur 10080 échoue lui aussi dans trois processus
séparés : les bilans CO₂ sont 65,834017, 66,032201 et 66,034192 kg.
Les traces complètes sont dans `candidate_fresh/audit.json`. Il s’agit d’un
contre-exemple conservé, pas d’une répétition supprimée des résultats.

`scripts/check_reproducibility.py` relance Python et SUMO dans des processus
séparés, avec des seeds de hachage Python différentes, puis compare les séries
complètes, les décisions, les bilans officiels et le protocole. Il termine en
erreur si une répétition diffère ; il ne masque pas cet écart par une moyenne.

## 2. Nouvelle récompense

Le mode `worst_interval` utilise exclusivement le CO₂, les NOx, le carburant
et les PM. Chaque mesure est divisée par un débit de référence obtenu avec
les feux classiques sur les seeds d’entraînement. Les quatre métriques ont
le même poids.

Pour chaque seconde, le coût est :

`0,5 × moyenne des quatre ratios + 0,5 × ratio maximal + moyenne des dépassements positifs au carré`.

Le dépassement positif vaut `max(ratio − 1, 0)`. La récompense est l’opposé
de la somme de ces coûts, y compris pendant les transitions jaune et rouge.
Cette fonction pénalise les pics et les déséquilibres ; elle ne garantit pas
mathématiquement une diminution de chaque bilan final.

Cette récompense ne dépend plus d’un cumul d’émissions absent des observations,
contrairement à l’ancien coût cumulatif. Le facteur d’actualisation est fixé
à 1 et la fin des 600 secondes est traitée comme une fin d’épisode, sans
estimation de récompenses futures au-delà de la fenêtre.

Le rythme d’apprentissage est réduit à `0,0001`. La tête du critique et la
mémoire de l’optimiseur sont réinitialisées, tout en conservant la politique
du modèle parent. Les normalisateurs des observations restent ceux du parent ;
les références utilisées par la nouvelle récompense sont recalibrées.

Les règles de sécurité restent actives. Les files et les temps d’attente
n’entrent pas dans la récompense. Les arrivées, départs, collisions et
téléportations servent à vérifier qu’un gain apparent ne provient pas d’un
blocage de la circulation.

## 3. Entraînement et sélection

- Parent : V5, checkpoint 1 536.
- Nouvelles seeds d’entraînement : 15, 16, 17.
- Validation : 1007, 1008, 1009, distinctes de l’entraînement.
- 2 048 nouvelles décisions PPO, soit 3 584 au compteur ; 12 nouveaux épisodes complets.
- Validation toutes les 512 décisions ; meilleur checkpoint retenu : 3 072.

Le checkpoint 2 560 réduit légèrement les quatre émissions en validation,
mais fait arriver trois véhicules de moins au total. Il n’est pas sélectionné.
Le checkpoint 3 072 conserve les arrivées mais ne réduit pas les quatre
émissions face à la référence. Il reste un candidat expérimental non déployé.

Le dossier du candidat est `runs/models/emission_duration_v7_20261022/`.
L’ancien modèle actif dans `runs/models/emission_ppo/` est conservé.

## 4. Essais finaux indépendants

Les seeds 10090, 10091 et 10092 sont réservées aux tests. Chaque méthode est
exécutée deux fois par seed, pendant 600 secondes : **12 simulations**.
L’ordre d’exécution des deux méthodes est alterné. Les conditions et le
protocole sont identiques dans chaque paire. Les bilans SUMO incluent les
véhicules qui n’ont pas terminé leur trajet.

| Métrique | Feux classiques : moyenne | RL : moyenne | Variation moyenne appariée |
|---|---:|---:|---:|
| CO₂ | 65,671996 kg | 65,576375 kg | −0,147 % |
| NOx | 23,042123 g | 23,007773 g | −0,150 % |
| Carburant | 21,290029 kg | 21,259030 kg | −0,147 % |
| PM | 4,854470 g | 4,862370 g | +0,164 % |

Une valeur négative indique une baisse par rapport aux feux classiques.
Les deux premières seeds donnent les mêmes résultats avec les deux méthodes.
La troisième diminue les trois premières métriques et augmente les PM.
Les répétitions de chaque méthode sont identiques sur ces trois seeds.
Il n’y a aucune collision ni téléportation, et aucune baisse des arrivées
ou départs dans les paires de test.

Ces essais ne démontrent ni quatre baisses, ni un gain général. La stabilité
observée sur ces trois seeds ne supprime pas le contre-exemple de l’audit
10080. Les petits pourcentages doivent donc être interprétés avec prudence.

## 5. Fichiers et reproduction

- `flowmind/env.py` : options et identification du simulateur.
- `flowmind/emission_rl.py` : observations et protections communes à l’apprentissage et à l’inférence.
- `scripts/train_emissions.py` : nouvelle récompense, reprise et sélection sur validation.
- `scripts/check_reproducibility.py` : audit exact dans des processus séparés.
- `scripts/compare_emission_trials.py` : comparaisons répétées, amplitudes observées et contrôle de circulation.
- `runs/reports/repro_v7_20261022/` : traces et audits, y compris les échecs.
- `runs/reports/duration_v7_20261022/` : essais, CSV, graphique et décision de non-activation.

Les dossiers `runs/` sont ignorés par Git. Les scripts et ce rapport permettent
de reproduire le protocole ; partager les checkpoints ou les résultats exige
de transmettre ces fichiers séparément.

```powershell
python scripts/check_reproducibility.py --controller fixed_time --seed 1005 --seconds 600 --count 3 --output runs/reports/nouvel_audit
python scripts/compare_emission_trials.py --model-dir runs/models/emission_duration_v7_20261022 --seeds 10100 10101 10102 --seconds 600 --repeats 3 --output runs/reports/nouvelle_comparaison
```

La documentation SUMO décrit la seed et le déterminisme attendu, sans constituer
une preuve de reproductibilité de notre installation :
[Randomness — documentation SUMO](https://sumo.dlr.de/docs/Simulation/Randomness.html).

## Vérification logicielle

La suite complète passe : **97 tests**. Elle couvre notamment la récompense,
les protections, les comparaisons à protocole identique et le rejet d’un gain
moyen qui masque une hausse entre répétitions ou une baisse des arrivées.
Ces tests ne remplacent pas les audits exécutés dans le vrai simulateur.
