"""
MODUŁ FUNKCJI POMOCNICZYCH
Funkcje pomocnicze, raportowanie i podsumowania
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

class ProjectUtils:
    """Klasa z funkcjami pomocniczymi dla projektu"""

    def __init__(self):
        self.project_name = "Analiza Czynników Diagnostycznych Zawału Serca"
        self.author = "Student"
        self.date = datetime.now().strftime("%Y-%m-%d")

    def generate_final_summary(self, hypothesis_results, df_analysis):

        print("\n" + "="*90)
        print("FINALNE PODSUMOWANIE PROJEKTU")
        print("="*90)

        print(f"\nINFORMACJE O PROJEKCIE:")
        print(f"   Nazwa: {self.project_name}")
        print(f"   Autor: {self.author}")
        print(f"   Data: {self.date}")
        print(f"   Liczba pacjentów: {len(df_analysis)}")
        print(f"   Liczba zmiennych: {len(df_analysis.columns)}")

        self._print_basic_statistics(df_analysis)

        self._print_hypothesis_summary(hypothesis_results)

        self._print_key_findings(hypothesis_results, df_analysis)

        self._print_recommendations()

        self._print_limitations()

        print("\n" + "="*90)
        print("KONIEC ANALIZY")
        print("="*90)

    def _print_basic_statistics(self, df):

        print(f"\nPODSTAWOWE STATYSTYKI DATASETU:")

        result_counts = df['Result'].value_counts()
        positive_pct = result_counts.get('Positive', 0) / len(df) * 100

        print(f"   • Przypadki zawału: {result_counts.get('Positive', 0)} ({positive_pct:.1f}%)")
        print(f"   • Przypadki bez zawału: {result_counts.get('Negative', 0)} ({100-positive_pct:.1f}%)")

        # Rozkład płci
        gender_counts = df['Gender'].value_counts()
        women_pct = gender_counts.get(0, 0) / len(df) * 100

        print(f"   • Kobiety: {gender_counts.get(0, 0)} ({women_pct:.1f}%)")
        print(f"   • Mężczyźni: {gender_counts.get(1, 0)} ({100-women_pct:.1f}%)")

        # Podstawowe parametry
        print(f"   • Średni wiek: {df['Age'].mean():.1f} ± {df['Age'].std():.1f} lat")
        print(f"   • Zakres wieku: {df['Age'].min()}-{df['Age'].max()} lat")
        print(f"   • Średnie tętno: {df['Heart rate'].mean():.1f} ± {df['Heart rate'].std():.1f} bpm")

        # Biomarkery
        print(f"   • Średnia troponina: {df['Troponin'].mean():.3f} ± {df['Troponin'].std():.3f} ng/mL")
        print(f"   • Średnie CK-MB: {df['CK-MB'].mean():.3f} ± {df['CK-MB'].std():.3f} ng/mL")

    def _print_hypothesis_summary(self, results):

        print(f"\nPODSUMOWANIE HIPOTEZ BADAWCZYCH:")

        hypotheses_info = {
            'h1': {
                'title': 'Wiek jako predyktor troponiny',
                'variables': 'Age, Gender, Heart rate → Troponin'
            },
            'h2': {
                'title': 'Płeć jako determinanta ciśnienia',
                'variables': 'Gender, Age, Blood sugar → Systolic BP'
            },
            'h3': {
                'title': 'Biomarkery jako predyktory zawału',
                'variables': 'Troponin, CK-MB, Heart rate → Result'
            }
        }

        confirmed_count = 0

        for hyp_id in ['h1', 'h2', 'h3']:
            if hyp_id in results:
                result = results[hyp_id]
                hyp_info = hypotheses_info[hyp_id]

                # Status hipotezy
                conclusion = result.get('hypothesis_conclusion', 'NIEZNANY')
                if 'POTWIERDZONA' in conclusion:
                    status_icon = "+"
                    confirmed_count += 1
                else:
                    status_icon = "-"

                print(f"\n   {status_icon} HIPOTEZA {hyp_id.upper()}: {hyp_info['title']}")
                print(f"      Zmienne: {hyp_info['variables']}")
                print(f"      Status: {conclusion}")
                print(f"      Uzasadnienie: {result.get('hypothesis_explanation', 'Brak')}")

                # Metryki specyficzne
                if result['type'] == 'regression':
                    r2 = result.get('r2_test', 0)
                    print(f"      R² = {r2:.3f} ({self._interpret_r2(r2)})")
                else:
                    auc = result.get('auc', 0)
                    accuracy = result.get('accuracy', 0)
                    print(f"      AUC = {auc:.3f}, Accuracy = {accuracy:.3f}")

        # Podsumowanie ogólne
        success_rate = confirmed_count / 3 * 100
        print(f"\n OGÓLNY WYNIK: {confirmed_count}/3 hipotez potwierdzonych ({success_rate:.1f}%)")

        if success_rate >= 66:
            overall_assessment = "BARDZO DOBRY - większość hipotez potwierdzona"
        elif success_rate >= 33:
            overall_assessment = "UMIARKOWANY - część hipotez potwierdzona"
        else:
            overall_assessment = "SŁABY - większość hipotez odrzucona"

        print(f"   🏆 OCENA OGÓLNA: {overall_assessment}")

    def _interpret_r2(self, r2):
        if r2 < 0.1:
            return "bardzo słaby wpływ"
        elif r2 < 0.3:
            return "słaby wpływ"
        elif r2 < 0.5:
            return "umiarkowany wpływ"
        elif r2 < 0.7:
            return "silny wpływ"
        else:
            return "bardzo silny wpływ"

    def _print_key_findings(self, results, df):

        print(f"\n💡 KLUCZOWE ODKRYCIA:")

        numeric_vars = ['Age', 'Gender', 'Heart rate', 'Systolic blood pressure',
                        'Blood sugar', 'CK-MB', 'Troponin', 'Result_Binary']
        corr_matrix = df[numeric_vars].corr()

        result_corrs = corr_matrix['Result_Binary'].abs().drop('Result_Binary').sort_values(ascending=False)
        strongest_predictor = result_corrs.index[0]
        strongest_corr = corr_matrix['Result_Binary'][strongest_predictor]

        print(f"   1. Najsilniejszy predyktor zawału: {strongest_predictor}")
        print(f"      Korelacja: r = {strongest_corr:.3f}")

        positive_group = df[df['Result'] == 'Positive']
        negative_group = df[df['Result'] == 'Negative']

        troponin_diff = positive_group['Troponin'].mean() - negative_group['Troponin'].mean()
        ckmb_diff = positive_group['CK-MB'].mean() - negative_group['CK-MB'].mean()

        print(f"   2. Różnice w biomarkerach (Pozytywne vs Negatywne):")
        print(f"      Troponina: +{troponin_diff:.3f} ng/mL ({troponin_diff/negative_group['Troponin'].mean()*100:+.1f}%)")
        print(f"      CK-MB: +{ckmb_diff:.3f} ng/mL ({ckmb_diff/negative_group['CK-MB'].mean()*100:+.1f}%)")

        men_heart_attack_rate = (df[(df['Gender'] == 1) & (df['Result'] == 'Positive')].shape[0] /
                                 df[df['Gender'] == 1].shape[0]) * 100
        women_heart_attack_rate = (df[(df['Gender'] == 0) & (df['Result'] == 'Positive')].shape[0] /
                                   df[df['Gender'] == 0].shape[0]) * 100

        print(f"   3. Różnice płciowe w występowaniu zawału:")
        print(f"      Mężczyźni: {men_heart_attack_rate:.1f}%")
        print(f"      Kobiety: {women_heart_attack_rate:.1f}%")
        print(f"      Różnica: {men_heart_attack_rate - women_heart_attack_rate:+.1f} punktów procentowych")

        # 4. Wiek i zawał
        mean_age_positive = positive_group['Age'].mean()
        mean_age_negative = negative_group['Age'].mean()
        age_diff = mean_age_positive - mean_age_negative

        print(f"   4. Wiek a zawał serca:")
        print(f"      Średni wiek z zawałem: {mean_age_positive:.1f} lat")
        print(f"      Średni wiek bez zawału: {mean_age_negative:.1f} lat")
        print(f"      Różnica: {age_diff:+.1f} lat")

        # 5. Najlepszy model predykcyjny
        if 'h3' in results and results['h3']['type'] == 'classification':
            best_auc = results['h3'].get('auc', 0)
            best_accuracy = results['h3'].get('accuracy', 0)
            print(f"   5. Najlepszy model predykcyjny (biomarkery):")
            print(f"      AUC: {best_auc:.3f} ({self._interpret_auc(best_auc)})")
            print(f"      Dokładność: {best_accuracy*100:.1f}%")

    def _interpret_auc(self, auc):
        if auc < 0.6:
            return "słaba zdolność predykcyjna"
        elif auc < 0.7:
            return "umiarkowana zdolność predykcyjna"
        elif auc < 0.8:
            return "dobra zdolność predykcyjna"
        elif auc < 0.9:
            return "bardzo dobra zdolność predykcyjna"
        else:
            return "doskonała zdolność predykcyjna"

    def _print_recommendations(self):

        print(f"\nREKOMENDACJE DLA PRZYSZŁYCH BADAŃ:")

        recommendations = [
            "Zwiększenie próby badawczej dla lepszej generalizowalności wyników",
            "Dodanie dodatkowych biomarkerów (np. NT-proBNP, D-dimer)",
            "Uwzględnienie historii medycznej pacjentów (choroby współistniejące)",
            "Analiza czasowa - kiedy wystąpiły pierwsze objawy",
            "Badanie wpływu leków na poziomy biomarkerów",
            "Walidacja modeli na niezależnej kohorcie pacjentów",
            "Analiza kosztów-korzyści różnych strategii diagnostycznych",
            "Badanie interakcji między zmiennymi (płeć × wiek, biomarkery × leki)"
        ]

        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")

    def _print_limitations(self):

        print(f"\n⚠OGRANICZENIA BADANIA:")

        limitations = [
            "Dane przekrojowe - brak możliwości wnioskowania o przyczynowości",
            "Ograniczona liczba zmiennych - mogą istnieć nieuwzględnione czynniki",
            "Brak informacji o czasie od wystąpienia objawów do pobrania próbek",
            "Możliwe błędy pomiarowe w laboratoryjnych oznaczeniach biomarkerów",
            "Brak stratyfikacji według wieku - różne normy dla różnych grup wiekowych",
            "Nieznane kryteria włączenia/wykluczenia pacjentów do badania",
            "Brak informacji o leczeniu przed pobraniem próbek krwi"
        ]

        for i, limitation in enumerate(limitations, 1):
            print(f"   {i}. {limitation}")

