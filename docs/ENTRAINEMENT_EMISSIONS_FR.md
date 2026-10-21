# Entraîner le modèle RL d’émissions

Trois contrôleurs sont disponibles : `fixed_time`, `rules` et `emission_model`.
Le dernier est une politique PPO centralisée qui choisit une phase par carrefour.
Il ne s’agit pas d’un LLM. Le modèle Flow d’Ahmed reste distinct et inchangé.

## Données et récompense

Les cartes et demandes viennent de `scenarios/`. SUMO génère les expériences
pendant l’apprentissage. Les entrées du réseau sont les émissions par phase
(CO₂, NOx, carburant, PM), la phase actuelle et la durée du vert. Les files et
l’attente n’entrent pas dans le réseau ou la récompense ; elles restent utilisées
par les contraintes de circulation (sorties saturées et prévention de famine).

La récompense est la négative de la moyenne pondérée des quatre quantités en mg
produites entre les décisions, divisées par quatre références fixes en mg/s.
Les secondes de transition sont incluses, sans division par la longueur de
l’intervalle. Les références sont estimées avec FixedTime exclusivement sur
les seeds d’entraînement, puis figées dans `manifest.json`. Le réseau reçoit
`log(1 + taux / référence)` pour chaque phase et chaque substance.

Le modèle entraîne tous les carrefours du scénario simultanément via une action
MultiDiscrete. Il est spécifique au réseau et à la disposition de ses phases.
Les contraintes sont appliquées par le même SafetyGate pendant l’apprentissage
et l’inférence : phases existantes, vert minimal/maximal, transition, saturation
et attente excessive. Les changements sont encore exécutés par env.py.
Le bouclier piéton complet du projet n’est pas implémenté par ce module.

Les nouveaux entraînements utilisent `emission_rl.safety` : vert minimal 10 s,
vert maximal 35 s et priorité aux directions en file au rouge depuis 30 s.
Une marge pour les décisions, le jaune et la libération du carrefour doit rester
avant le seuil de téléportation SUMO de 60 s, qui n'est pas augmenté. Cela limite
la famine aux feux sans garantir l'absence de tout blocage physique en aval.
Le cycle classique de 30 s est conservé. Les protections sont enregistrées avec
le modèle et utilisées aussi à l'inférence ; le dashboard peut les renforcer,
mais pas les relâcher au-delà des limites entraînées.

## Installer et entraîner

```powershell
python -m pip install -r requirements.txt
python scripts/train_emissions.py --seconds 600 --timesteps 20000 --train-seeds 0 1 2 --validation-seeds 1000 1001 --output runs/models/emission_ppo
```

Arrêter les simulations du dashboard avant l’entraînement. Ne pas lancer plusieurs
entraînements ou évaluations simultanément dans la même session Python.
Le dossier de sortie doit être vide : aucun modèle existant n’est écrasé.

Les poids se définissent par `--weights 1 1 1 2` (CO₂, NOx, carburant, PM).
Ce sont aussi les valeurs par défaut de `emission_rl.weights` : les PM comptent
deux fois après normalisation. Les anciens modèles gardent les poids enregistrés
dans leur manifeste tant qu'ils ne sont pas remplacés.
Ils sont fixes pendant cet entraînement. Les poids du formulaire dashboard
pilotent les règles Python ; ils ne réentraînent pas le modèle chargé.

Validation sur des seeds distinctes, tous les 2048 pas par défaut
(`--validation-every`). La sélection minimise d'abord les collisions, puis les
téléportations sur toute la validation, puis le coût pondéré des émissions.
Ainsi une meilleure récompense avec davantage de blocages ne remplace pas
le checkpoint retenu. Ce checkpoint est enregistré dans `best.zip`.
`last.zip` conserve l’état final, et les CSV
conservent les récompenses par épisode et par validation. PPO travaille par
rollouts de 64 pas ; le nombre réel de pas peut dépasser la demande.
Pendant une reprise, le parent est aussi évalué avec les nouvelles protections
et les nouveaux poids avant l’apprentissage : il reste le checkpoint retenu si
aucun nouveau checkpoint ne le dépasse selon ces critères. Les diagnostics de
validation incluent les téléportations, collisions, arrivées et émissions TraCI.
Ces dernières servent au suivi ; les comparaisons finales utilisent les totaux
officiels de tripinfo.xml, qui incluent les trajets inachevés.
`best_timesteps` indique l’étape du checkpoint retenu, `actual_timesteps` le total
exécuté. `progress.json` permet de suivre l’avancement entre deux validations.

Les premiers modèles très courts portent la mention `smoke_test`. Le modèle
fourni pour vérifier le parcours n’est pas une preuve de performance. Le nombre
de pas, les seeds, poids, références, réseau et paramètres sont dans le manifeste.
Les durées courtes n’observent pas tous les effets de circulation ; utiliser
des durées et demandes variées pour l’étude finale. Faire varier la seed ne
change pas nécessairement la demande prédéfinie du fichier de routes.

## Fine-tuning

```powershell
python scripts/train_emissions.py --resume runs/models/emission_ppo --output runs/models/emission_ppo_v2 --seconds 600 --timesteps 20000 --train-seeds 3 4 5 --validation-seeds 1002 1003
```

Le modèle parent est chargé et entraîné davantage ; les références de
normalisation sont conservées. Le réseau doit rester compatible. Les seeds de tous les modèles parents restent réservées : une ancienne seed
de validation ne peut pas devenir une seed d’entraînement. Le parent
reste préservé. Pour utiliser une nouvelle version dans le dashboard, conserver
une sauvegarde de la version active puis placer le dossier validé à l’emplacement
`runs/models/emission_ppo`. Ne pas remplacer les fichiers pendant une simulation.

## Comparaison finale sans apprentissage

```powershell
python scripts/evaluate.py --controllers fixed_time rules emission_model --seconds 600 --seeds 10000 10001 10002
```

Pour tester une nouvelle version sans remplacer le modèle actif du dashboard :

```powershell
python scripts/evaluate.py --controllers fixed_time rules emission_model --model-dir runs/models/emission_ppo_v2 --seconds 600 --seeds 10000 10001 10002
```

Le dossier du modèle et son empreinte sont enregistrés avec chaque expérience.

Réserver ces seeds de test : elles ne doivent jamais servir au réglage des règles,
de la récompense ou du modèle. Les seeds d’entraînement et de validation sont
refusées pour une évaluation finale du modèle. Garder mêmes carte, routes, durée,
seed et protections pour chaque paire. Le modèle est chargé en inférence
déterministe et ne s’entraîne pas durant la comparaison.

Vérifier les quatre émissions, les arrivées, les véhicules restants, les collisions
et téléportations. Une politique peut émettre moins tout en bloquant les véhicules,
ou améliorer le CO₂ en dégradant les PM. Aucune amélioration n’est garantie.
Comparer les totaux officiels SUMO enregistrés dans result.json et les CSV exportés.

## Dashboard

`python scripts/dashboard.py` ouvre l’interface locale. La méthode **Agent avec
modèle RL** est activée si le checkpoint et son manifeste existent pour le scénario
choisi. Aucun retour silencieux à un contrôleur classique si le modèle manque.
L’historique contient un sélecteur d’essai pour chaque méthode ; chaque courbe
permet les trois tracés ou un tracé individuel (modèle en violet).
