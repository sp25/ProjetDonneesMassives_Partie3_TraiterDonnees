from unittest import TestCase
from traiter_donnees import *

class test_explorer_donnees(TestCase):

    # Exécution de l'algo final
    def test_explorer_random_forest_predire_casual_et_registered(self):
        test = explorer_random_forest_predire_casual_et_registered(30)
        self.assertEqual(1, 1)

    # Exécution de l'algo sur le jeu de données test pour soumission à Kaggle.
    def test_soumettre_kaggle_casual_et_registered(self):
        test = soumettreKaggle_casual_et_registered()
        self.assertEqual(1, 1)




    def test_explorer_random_forest_predire_casual_et_registered_sans_log(self):
        test = explorer_random_forest_predire_casual_et_registered_sans_log()
        self.assertEqual(1, 1)

    def test_explorer_random_forest_predire_uniquement_count_sans_log(self):
        test = explorer_random_forest_predire_uniquement_count()
        self.assertEqual(1, 1)

    def test_explorer_random_forest_predire_uniquement_count_AVEC_log(self):
        test = explorer_random_forest_predire_uniquement_count_AVEC_log()
        self.assertEqual(1, 1)

    def test_extraire_mauvaises_predictions(self):
        test = extraireDansCsvResultsPiresQueMoyenne()
        self.assertEqual(1, 1)

    def test_explorer_nbrArbre_randomForest(self):
        test = explorer_nbrArbre_randomForest()
        self.assertEqual(1, 1)

    def test_explorer_profondeurMax_randomForest(self):
        test = explorer_profondeurMax_randomForest()
        self.assertEqual(1, 1)

    def test_explorer_maxFeature_randomForest(self):
        test = explorer_maxFeature_randomForest()
        self.assertEqual(1, 1)

    def test_explorer_algorithmes(self):
        test = explorer_algorithmes()
        self.assertEqual(1, 1)