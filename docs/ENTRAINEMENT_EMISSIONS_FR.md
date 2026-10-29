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
Le vert classique est de 20 s depuis le 29 octobre 2026 (30 s avant). Les protections sont enregistrées avec
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
téléportations sur toute la validation. Les nouveaux entraînements établissent
aussi une référence FixedTime sur ces mêmes seeds de validation. Ils comparent
les quatre totaux officiels de `tripinfo.xml` par paire, puis minimisent la plus
forte dégradation relative aux feux classiques. Une baisse des arrivées ou des
départs est prise en compte dans cette dégradation pour éviter un gain apparent
obtenu en retenant les véhicules. Le coût environnemental pondéré départage les
checkpoints à égalité. Ainsi une meilleure récompense avec davantage de blocages ne remplace pas
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

Dans `validation.csv`, les colonnes `*_mg` sont les anciennes sommes TraCI de
suivi, tandis que `co2_ratio`, `nox_ratio`, `fuel_ratio` et `pm_ratio` reposent sur
les totaux officiels SUMO. Les ratios sont moyennés sur les paires de seeds :
1 signifie égalité avec FixedTime, moins de 1 une baisse, plus de 1 une hausse.
`all_emissions_lower` exige les quatre ratios sous 1. `worst_ratio` inclut aussi
les éventuels déficits d'arrivées et de départs ; son minimum est 1.
Les références et XML utilisés sont conservés dans `validation_baseline.json`,
`validation_baseline/` et `validation_trials/`. Ces données de validation ne
servent ni à apprendre les paramètres PPO ni à normaliser ses observations.
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
sans apprentissage durant la comparaison. Les anciens modèles et les manifestes
sans `deterministic_inference` conservent le choix de la phase la plus probable.
Un modèle peut aussi porter `deterministic_inference: false` : il tire alors les
phases selon les probabilités apprises par PPO. Ce n'est pas un contrôleur
uniformément aléatoire ; les probabilités dépendent du modèle et de ses entrées.
La seed de cette politique est la seed SUMO + `policy_seed_offset` (100000 par
défaut). Elle est enregistrée dans les résultats pour reproduire les essais.
Ce mode doit être choisi sur la validation avant les tests, et reste soumis au
même SafetyGate. Le dashboard indique le mode retenu dans le manifeste.

La commande de diagnostic suivante évalue les probabilités apprises sur les
seeds de validation, avec les mêmes références officielles que l'entraînement :

```powershell
python scripts/validate_emission_sampling.py --model-dir runs/models/emission_ppo_v4_20261021 --checkpoint last --baseline runs/models/emission_ppo_v4_20261021/validation_baseline.json --output runs/models/emission_ppo_v4_20261021/sampling_validation_manual --seeds 1002 1003
```

Le dossier de sortie doit être nouveau. La référence doit correspondre au même
scénario et à la même durée que le manifeste. Le diagnostic ne modifie pas le
modèle : il écrit les mesures et ratios dans `result.json`. L'entraînement actuel
valide ses checkpoints avec le choix le plus probable ; ce diagnostic permet de
comparer ensuite un mode probabiliste sans utiliser les seeds de test. Les
tirages ne doivent pas être répétés avec différentes seeds pour sélectionner
uniquement les résultats favorables.

Vérifier les quatre émissions, les arrivées, les véhicules restants, les collisions
et téléportations. Une politique peut émettre moins tout en bloquant les véhicules,
ou améliorer le CO₂ en dégradant les PM. Aucune amélioration n’est garantie.
Comparer les totaux officiels SUMO enregistrés dans result.json et les CSV exportés.

## Dashboard

### Expérience de durées de vert et récompense équilibrée (22 octobre)

Après un essai montrant 264 changements de phase pour l'ancien RL contre 153
pour la référence, une nouvelle variante est testée séparément :

```powershell
python scripts/train_emissions.py --output runs/models/emission_duration_v5_20261022 --seconds 600 --timesteps 2048 --validation-every 512 --train-seeds 9 10 --validation-seeds 1004 1005 --weights 1 1 1 2 --action-mode duration --reward-mode balanced
```

`duration` encode les actions comme un seuil de vert de 20, 25, 30 ou 35 secondes,
réévalué à chaque décision. L'ordre cyclique des phases est conservé. Ce n'est pas
une durée verrouillée au début du vert : les observations suivantes peuvent
ajuster le seuil. Toutes les demandes passent ensuite dans le SafetyGate.
Les premiers paramètres privilégient 30 secondes ; cette initialisation n'est
pas un apprentissage ni une preuve d'amélioration. Les anciennes politiques
continuent d'interpréter leurs actions comme des numéros de phase.

`balanced` utilise une différence de coût cumulatif : moitié moyenne pondérée,
moitié maximum des quatre émissions normalisées. Les poids de la moyenne
restent 1 / 1 / 1 / 2. Les normalisations sont calculées uniquement sur les seeds
d'entraînement et restent fixes. La somme non actualisée des récompenses est
l'opposé du coût final (à la première seconde commune près). Cela limite les
compensations entre polluants sans garantir quatre réductions simultanées.
Attente, files et comptages ne sont pas ajoutés à ce coût ; ils restent des
contraintes de sécurité ou des vérifications du débit en validation.

Cette variante repart d'un réseau neuf car le sens et le nombre d'actions
changent. Ne pas réutiliser les poids d'une politique de phases dans cette
architecture. Le manifeste conserve les modes, durées, poids et seeds. Le modèle
actif est conservé pendant l'expérience et n'est pas remplacé par l'entraînement.

Bilan de cette campagne : 2 048 étapes exécutées, checkpoint 1 536 retenu sur
la validation, puis testé sans apprentissage sur 10060, 10061, 10062 (600 s).
Variations des moyennes par rapport aux feux classiques : CO₂ -0,01724 %, NOx
-0,01613 %, carburant -0,01724 %, PM -0,03222 %. Deux essais sont exactement
identiques à la référence ; seul le troisième apporte la petite baisse.
Les arrivées sont identiques et aucun des six essais ne produit de collision
ou de téléportation. Cela ne prouve pas une supériorité générale.

Après sélection en validation et vérification de sécurité des tests, la variante
a été activée dans `runs/models/emission_ppo/`. L'ancien modèle est conservé
dans `runs/models/emission_ppo_before_duration_v5_20261022/`. Le bilan complet
et les identifiants sont dans `runs/reports/duration_v5_20261022/rapport.md`.

### Reprise V6 : davantage d'étapes ne suffit pas

La campagne suivante reprend le checkpoint 1 536, ajoute 4 096 étapes sur
les seeds 11, 12, 13, 14 et valide sur 1004, 1005, 1006. Elle termine 28
épisodes complets et atteint le compteur 5 632. Les sauvegardes 3 584 et
5 632 sont moins bonnes en validation ; le checkpoint 1 536 reste retenu.

Le dernier état (5 632) a été testé séparément à titre de diagnostic, après
son rejet en validation, sur 10070, 10071, 10072. Les classiques sont répétés
juste avant chaque essai : CO₂ +3,194 %, NOx +3,474 %, carburant +3,194 %,
PM -1,490 %. Il fait arriver moins de véhicules, sans collision ni
téléportation. Il n'a pas été activé. Le dashboard marque ces essais
`diagnostic non déployé` ; le modèle actif précédent reste inchangé.

Un contrôle de répétition de la référence classique sur 1004 a aussi produit
des bilans différents à paramètres identiques ; la cause n'est pas isolée.
Les faibles baisses de V5 ne constituent donc pas une preuve fiable de gain
général. Les bilans et répétitions sont conservés dans
`runs/reports/duration_v6_20261022/`, avec `rapport.md`, les CSV, le graphique
et la décision de non-activation. Il faut consolider la reproductibilité et
le compromis de récompense avant d'interpréter de petits pourcentages.

`python scripts/dashboard.py` ouvre l’interface locale. La méthode **Agent avec
modèle RL** est activée si le checkpoint et son manifeste existent pour le scénario
choisi. Aucun retour silencieux à un contrôleur classique si le modèle manque.
L’historique contient un sélecteur d’essai pour chaque méthode ; chaque courbe
permet les trois tracés ou un tracé individuel (modèle en violet).


## V7 — audit de reproductibilité et récompense équilibrée (22 octobre 2026)

Le mode `worst_interval` donne le même poids aux quatre émissions, pénalise
la pire émission normalisée et les dépassements au carré. Il utilise les
références d’entraînement uniquement, un gamma de 1 et un apprentissage à
0,0001. La circulation reste un critère de validation, sans entrer dans la
récompense. `--adapt-reward` autorise explicitement cette modification lors
d’une reprise compatible des actions.

Après 2 048 nouvelles décisions, les 12 simulations de test (3 seeds, deux
répétitions par méthode, 600 s) donnent en moyenne : CO₂ −0,147 %, NOx −0,150 %,
carburant −0,147 %, PM +0,164 %. Le nouveau checkpoint n’est pas activé.
Le protocole ballistic stabilise certains cas mais des écarts subsistent
sur d’autres seeds : la reproductibilité générale n’est pas garantie.
Les audits exacts échouent explicitement en présence d’un écart.

Voir [le rapport détaillé V7](RAPPORT_REPRODUCTIBILITE_RL_V7_FR.md) pour les
conditions, les limites, les commandes et les fichiers de résultats.
