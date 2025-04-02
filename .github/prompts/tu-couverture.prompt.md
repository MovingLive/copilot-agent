Exécuter les tests unitaires en utilisant l'outil de test intégré de VSCode (le tool run_tests) pour cibler identifier la couverture de code et augmenter l'augmenter à minimum 80%.

Requis pour de bons tests unitaires:

- regarder les schémas pydantics attendus des endpoints pour s'assurer d'envoyer l'ensemble des information attendues
- utiliser les factory situées dans `app/tests/mock_generators` pour générer des données de tests
- utiliser les mocks pour simuler des appels à des services externes
- utiliser les fixtures pour partager des données entre les tests
- utiliser les paramétrages pour tester des cas de bords
