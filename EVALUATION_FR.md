# Évaluer FlowMind

## Score environnemental depuis le 21 octobre 2026

L’agent `rules` utilise exclusivement quatre mesures dans son score : CO₂,
NOx, carburant et PM. Elles sont lues sur les voies entrantes via TraCI en mg/s,
puis regroupées par phase. Pour chaque mesure, la somme sur les voies servies
est divisée par la somme de cette mesure sur toutes les voies entrantes du
carrefour. Une somme nulle contribue zéro. Le score est la moyenne pondérée
de ces quatre parts ; les poids sont positifs ou nuls et au moins un doit être
strictement positif. Ils ne sont pas des pourcentages.

L’agent favorise la phase autorisée ayant la charge environnementale actuelle
la plus élevée. Il s’agit d’une règle réactive de priorité aux émissions, pas
d’une prédiction de réduction : une accélération peut augmenter les émissions.
La diminution du CO₂, des NOx, du carburant ou des PM doit être mesurée sur des
essais comparables, et n’est pas garantie.

Les paramètres `co2_weight`, `nox_weight`, `fuel_weight`, `pm_weight` remplacent
les anciens poids de files et d’attente dans config.yaml et le dashboard.
La marge de changement est maintenant dimensionnelle zéro, entre 0 et 1
(valeur initiale 0,05). Les paramètres de sécurité restent séparés : vert
minimal/maximal, protection contre une attente excessive et sorties saturées.
Les files, l’attente et l’occupation ne contribuent pas au score. La protection
contre une attente excessive conserve néanmoins son suivi des voies en attente.

Les structures partagées sont étendues de manière compatible : `LaneEmissions`
et le champ optionnel `JunctionState.lane_emissions`. Les autres agents peuvent
ignorer ce champ ; `rules` refuse des mesures manquantes plutôt que d’inventer
un score. Les phases demandées et le score retenu figurent dans decisions.csv.
Les empreintes du protocole distinguent les nouvelles expériences des anciennes.

Les passages plus bas et les rapports datés d’avant cette évolution décrivant
un score fondé sur les files correspondent à l’ancienne version.

## Pages Simulation et Historique

La page **Simulation** lance un seul essai, avec l’agent à règles ou sans agent
(contrôleur FixedTime). Choisir la méthode, le scénario, une durée et une seed.
Pendant l’essai, modifier la méthode et/ou les règles puis cliquer sur
**Appliquer pendant la simulation**. Le changement est appliqué au prochain point
de décision, sans recommencer le trafic. Le vert minimal et les transitions restent
respectés. La fenêtre SUMO montre le comportement réel des feux et des véhicules.
Les métriques futures peuvent changer ; les mesures déjà enregistrées sont conservées.

Les décisions et leurs révisions sont inscrites dans timeseries.csv et les paramètres
de chaque révision dans control_events.json. Les essais modifiés sont étiquetés
`interactive` dans result.json et exclus des comparaisons automatiques de contrôleurs
fixes : le trafic garde la mémoire des méthodes précédentes.

La page **Historique des métriques** propose un essai avec agent et un essai sans
agent, un début et une fin de visualisation en secondes simulées. Chaque métrique
possède sa propre courbe et un sélecteur **Avec agent / Sans agent / Les deux**.
On peut choisir le même essai interactif dans les deux listes pour voir ses périodes
de chaque méthode. Les lignes ne relient pas des segments de méthodes différentes.
Les conditions différentes sont signalées. Pour une comparaison contrôlée, lancer
deux essais séparés avec les mêmes scénario, durée, seed et paramètres.

La courbe d’attente montre le temps d’arrêt courant moyen des véhicules présents,
pas l’attente totale par trajet calculée à la fin. Les anciens essais sans cette
mesure affichent une absence de données. Chaque seconde est conservée en CSV ;
la période est filtrée avant de limiter le tracé à 600 points pour le navigateur.

## Démarrer le dashboard

Double-cliquez sur `Lancer-dashboard.cmd` ou lancez :

```powershell
python scripts/dashboard.py
```

Le navigateur ouvre http://127.0.0.1:8501. Le serveur reste ouvert dans le terminal.
Choisissez le scénario, la durée et les seeds, puis les contrôleurs à comparer.
Cochez **Ouvrir la fenêtre SUMO** pour voir la circulation. Cliquez sur **Démarrer**.
Chaque contrôleur est exécuté successivement pour chaque seed, avec les mêmes entrées.
**Arrêter** conserve les résultats partiels et les exclut des comparaisons.
L’état et les courbes sont actualisés toutes les deux secondes, les instantanés toutes les cinq secondes simulées.

La section **Évolution des métriques en temps réel** affiche le trafic présent et arrêté,
la file moyenne par voie, la vitesse moyenne instantanée (km/h) et le débit de CO₂ (g/s).
Les courbes suivent automatiquement chaque nouvel essai (contrôleur et seed) ;
le sélecteur permet aussi de consulter un essai enregistré. Elles restent disponibles
après un arrêt ou la fin de la simulation. Jusqu’à 600 points régulièrement espacés
sont affichés, en conservant le premier et le dernier ; le téléchargement CSV
conserve toutes les mesures à chaque seconde. La vitesse est indisponible quand
le réseau est vide. Le débit instantané de CO₂ est distinct du total final en kg.

Les seeds par défaut sont 100 et 101 pour découvrir. Pour l’évaluation prévue dans le
guide, saisissez `100,101,102,103,104,105,106,107,108,109`. Une durée d’au moins
3600 secondes est préférable aux essais courts de vérification. Choisissez une
fenêtre adaptée aux départs du scénario ; elle peut couper des trajets.

## Contrôleurs disponibles

- **Agent optimiseur · règles Python** (`rules`) : une instance par carrefour,
  sans modèle ni entraînement. Le score favorise les files et l’attente, prend
  en compte les véhicules en approche et pénalise les files en sortie. Une marge
  évite les changements pour un faible gain ; le vert minimal et maximal et une
  priorité après une longue attente limitent les oscillations et la famine.
  Les coefficients se trouvent dans la section `rules` de config.yaml.
  Réduire les arrêts et redémarrages peut réduire les émissions, mais l’agent
  ne prédit pas le CO₂ : seules les mesures comparatives permettent de le vérifier.
  Ce contrôleur ne remplace pas un dispositif de sécurité pour les piétons.

```powershell
python scripts/evaluate_emissions.py --controllers fixed_time rules --seconds 300 --seeds 100 101
python scripts/run.py --controller rules --gui
```

- **Feux classiques** : classe FixedTime du projet, durée verte configurée dans config.yaml ;
  ce contrôleur passe par le même environnement et ses protections que les autres.
  Il ne s’agit pas du programme natif SUMO laissé sans intervention Python.
- **MaxPressure** : règles basées sur les files amont et aval.
- **Flow · règles** : règles de secours de l’agent Flow, explicitement sans modèle.
- **Flow · modèle entraîné** : activé seulement si runs/models/flow_ppo.zip existe ;
  l’expérience échoue si le modèle ne se charge pas. L’agent peut néanmoins revenir
  à ses règles en cours de simulation ; consulter decisions.csv.

Les agents bus, urgence, piétons, incident et l’orchestrateur restent des squelettes
dans cette branche. Ils ne sont pas présentés comme fonctionnels dans le dashboard.
Les scénarios proposés utilisent le trafic fourni, sans injection d’accident,
d’ambulance, de bus ni de piéton. Ajouter et valider ces scénarios avec leurs
responsables sera nécessaire avant de mesurer ces cas particuliers. L’attente
d’un bus ne doit pas être confondue avec son retard par rapport à un horaire.

## Données enregistrées

Chaque essai crée `runs/<identifiant>/` avec :

- metadata.json : scénario, contrôleur, seed, durée, copie de configuration et empreintes ;
- tripinfo.xml : sorties SUMO par trajet, trajets inachevés inclus ;
- statistics.xml : statistiques globales SUMO ;
- timeseries.csv : mesures à chaque seconde, y compris pendant l’orange et le rouge ;
- decisions.csv : phases demandées et appliquées, explications de l’agent ;
- live.json : dernier état pour le dashboard ;
- result.json : métriques finales et état (completed, stopped ou failed).

`runs/reports/metrics.csv` rassemble les expériences ; `comparison.csv` contient les
comparaisons ; les PNG présentent l’attente, les files et le CO₂. Le dashboard
dessine aussi ses propres graphiques depuis les données calculées, sans lire les PNG.
Ces résultats sont ignorés par Git. Le code est conservé dans la branche manoubi.

## Définition des métriques

| Mesure | Définition |
|---|---|
| Attente moyenne / maximale (s) | waitingTime SUMO sur tous les véhicules partis, y compris l’attente observée des trajets inachevés ; ne comprend pas l’attente avant insertion |
| Durée moyenne des trajets (s) | duration des seuls trajets terminés ; indisponible si aucune arrivée |
| File moyenne par voie | moyenne temporelle des véhicules arrêtés sur toutes les voies non internes du réseau, voies vides incluses |
| File maximale | maximum observé sur une voie à une seconde donnée |
| Vitesse moyenne (km/h) | moyenne pondérée par les véhicules présents à chaque seconde |
| CO₂ (kg), NOx/particules (g), carburant (kg) | sommes officielles SUMO dans tripinfo.xml, trajets inachevés inclus ; modèle d’émissions du scénario. Les taux instantanés TraCI sont aussi conservés pour inspection |
| CO₂ par véhicule parti | CO₂ de la fenêtre divisé par le nombre de départs effectifs, pas une empreinte de trajet complet |
| Arrivés / partis / actifs / en attente d’insertion | indicateurs à examiner ensemble pour éviter une amélioration artificielle due aux trajets inachevés |
| Événements de véhicules en collision | somme des nombres de véhicules impliqués signalés à chaque seconde ; pas un nombre de collisions distinctes |
| Téléportations | nombre de débuts de téléportation, distinct des collisions |

La fenêtre a une durée exacte identique entre contrôleurs, même si une transition
de feu est en cours à sa fin. Les mesures ne couvrent pas une simulation infinie.

## Comparaisons et conclusions

Une référence FixedTime est appariée à chaque autre contrôleur sur : scénario,
situation, durée, seed, empreinte des entrées et du protocole (configuration et code).
Un modèle entraîné est identifié par son empreinte. Les répétitions identiques
utilisent la dernière expérience terminée, sans compter deux fois la même seed.
Les essais interrompus et échoués restent visibles mais sont exclus.

Pour chaque indicateur : variation (%) = 100 × (candidat − référence) / référence.
Si la référence vaut zéro, le pourcentage est indéfini. La dispersion affichée est
l’écart-type des différences appariées entre seeds. Aucun test de significativité
n’est réalisé. Les conclusions automatiques décrivent les observations, sans LLM
et sans promettre qu’un système est toujours meilleur.

Avec des itinéraires prédéfinis, changer la seed ne crée pas forcément une nouvelle
demande : les heures de départ restent identiques, seuls les éléments aléatoires
activés dans SUMO changent. Pour étudier différents niveaux de trafic, préparer
des demandes distinctes et conserver les mêmes entrées dans chaque paire.

## Utilisation sans interface

```powershell
python scripts/evaluate.py --seconds 600 --seeds 100 101 --controllers fixed_time max_pressure flow_heuristic
python -m pytest
```

Sans `--seeds`, evaluate.py utilise les dix seeds de config.yaml. Ajouter `--gui`
pour ouvrir SUMO. Les scripts historiques run.py et train.py restent utilisables.
Ne pas lancer plusieurs évaluations simultanément : elles partagent la connexion
TraCI par défaut. Le dashboard refuse lui-même un second lancement concurrent.
