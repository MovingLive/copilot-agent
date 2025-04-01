Pour résoudre les problèmes de pertinence, je vous recommande de vérifier :

* Si les distances sont trop élevées (généralement >0.6 indique une faible similarité)
* Si le nombre de résultats valides correspond à vos attentes
* Si la segmentation de vos documents est appropriée (les segments trop courts ou trop longs peuvent affecter la qualité)

Vous pouvez également essayer d'augmenter le paramètre `precision_priority` à `True` lors de l'appel à `retrieve_similar_documents` pour privilégier la précision à la vitesse.
