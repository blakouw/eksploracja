
import sys
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader
from descriptive_stats import DescriptiveStats
from advanced_analysis import AdvancedAnalysis
from hypothesis_testing import HypothesisTesting
from visualizations import Visualizations
from utils import ProjectUtils

def main():

    print("=" * 90)
    print("ANALIZA MEDYCZNA - EKSPLORACJA DANYCH Z TRZEMA HIPOTEZAMI BADAWCZYMI")
    print("=" * 90)

    # 1. Wczytanie i przygotowanie danych
    print("\n ETAP 1: Wczytywanie i przygotowanie danych...")
    data_loader = DataLoader()
    df, df_analysis = data_loader.load_and_prepare_data('Medicaldataset.csv')

    # 2. Statystyki opisowe
    print("\n ETAP 2: Analiza statystyk opisowych...")
    desc_stats = DescriptiveStats(df_analysis)
    desc_stats.run_all_analyses()

    # 3. Zaawansowana analiza
    print("\n ETAP 3: Zaawansowana analiza danych...")
    adv_analysis = AdvancedAnalysis(df_analysis)
    adv_analysis.run_all_analyses()

    # 4. Testowanie hipotez
    print("\n ETAP 4: Testowanie hipotez badawczych...")
    hyp_testing = HypothesisTesting(df_analysis)
    results = hyp_testing.test_all_hypotheses()

    # 5. Drzewa decyzyjne (CZĘŚĆ III)
    print("\n ETAP 5: Indukcja drzew decyzyjnych...")
    from decision_trees import run_decision_tree_analysis
    tree_analysis = run_decision_tree_analysis(df_analysis)

    # 6. Generowanie wizualizacji
    print("\n ETAP 6: Generowanie wizualizacji...")
    viz = Visualizations(df_analysis)
    viz.create_all_plots()

    # 7. Podsumowanie wyników
    print("\n ETAP 7: Podsumowanie wyników...")
    utils = ProjectUtils()
    utils.generate_final_summary(results, df_analysis)

    print("\n ANALIZA ZAKOŃCZONA POMYŚLNIE!")
    print("Wszystkie wyniki zostały wygenerowane i wyświetlone.")

if __name__ == "__main__":
    main()