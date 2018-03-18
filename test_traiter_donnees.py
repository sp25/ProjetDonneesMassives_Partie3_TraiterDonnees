from unittest import TestCase
from traiter_donnees import explorer_algorithmes
from traiter_donnees import explorer_random_forest_predire_uniquement_count
from traiter_donnees import explorer_random_forest_predire_casual_et_registered
from traiter_donnees import explorer_nbrArbre_randomForest
from traiter_donnees import explorer_profondeurMax_randomForest
from traiter_donnees import explorer_maxFeature_randomForest

class test_explorer_donnees(TestCase):
    def test_explorer_algorithmes(self):
        test = explorer_algorithmes()
        self.assertEqual(1, 1)

    def test_explorer_random_forest_predire_uniquement_count(self):
        test = explorer_random_forest_predire_uniquement_count()
        self.assertEqual(1, 1)

    def test_explorer_random_forest_predire_casual_et_registered(self):
        test = explorer_random_forest_predire_casual_et_registered()
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

