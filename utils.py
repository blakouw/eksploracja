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
        """Generuje finalne podsumowanie całego projektu"""

        print("\n" + "="*90)
        print("FINALNE PODSUMOWANIE PROJEKTU")
        print("="*90)

        # Informacje podstawowe
        print(f"\n📋 INFORMACJE O PROJEKCIE:")
        print(f"   Nazwa: {self.project_name}")
        print(f"   Autor: {self.author}")
        print(f"   Data: {self.date}")
        print(f"   Liczba pacjentów: {len(df_analysis)}")
        print(f"   Liczba zmiennych: {len(df_analysis.columns)}")

        # Statystyki podstawowe
        self._print_basic_statistics(df_analysis)

        # Podsumowanie hipotez
        self._print_hypothesis_summary(hypothesis_results)

        # Kluczowe odkrycia
        self._print_key_findings(hypothesis_results, df_analysis)

        # Rekomendacje
        self._print_recommendations()

        # Ograniczenia badania
        self._print_limitations()

        print("\n" + "="*90)
        print("KONIEC ANALIZY")
        print("="*90)

    def _print_basic_statistics(self, df):
        """Wyświetla podstawowe statystyki"""

        print(f"\n📊 PODSTAWOWE STATYSTYKI DATASETU:")

        # Rozkład wyniku
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
        """Wyświetla podsumowanie wszystkich hipotez"""

        print(f"\n🎯 PODSUMOWANIE HIPOTEZ BADAWCZYCH:")

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
                    status_icon = "✅"
                    confirmed_count += 1
                else:
                    status_icon = "❌"

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
        print(f"\n   📈 OGÓLNY WYNIK: {confirmed_count}/3 hipotez potwierdzonych ({success_rate:.1f}%)")

        if success_rate >= 66:
            overall_assessment = "BARDZO DOBRY - większość hipotez potwierdzona"
        elif success_rate >= 33:
            overall_assessment = "UMIARKOWANY - część hipotez potwierdzona"
        else:
            overall_assessment = "SŁABY - większość hipotez odrzucona"

        print(f"   🏆 OCENA OGÓLNA: {overall_assessment}")

    def _interpret_r2(self, r2):
        """Interpretuje wartość R²"""
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
        """Wyświetla kluczowe odkrycia"""

        print(f"\n💡 KLUCZOWE ODKRYCIA:")

        # 1. Najsilniejsze korelacje
        numeric_vars = ['Age', 'Gender', 'Heart rate', 'Systolic blood pressure',
                        'Blood sugar', 'CK-MB', 'Troponin', 'Result_Binary']
        corr_matrix = df[numeric_vars].corr()

        # Znajdź najsilniejszą korelację z wynikiem
        result_corrs = corr_matrix['Result_Binary'].abs().drop('Result_Binary').sort_values(ascending=False)
        strongest_predictor = result_corrs.index[0]
        strongest_corr = corr_matrix['Result_Binary'][strongest_predictor]

        print(f"   1. Najsilniejszy predyktor zawału: {strongest_predictor}")
        print(f"      Korelacja: r = {strongest_corr:.3f}")

        # 2. Różnice między grupami
        positive_group = df[df['Result'] == 'Positive']
        negative_group = df[df['Result'] == 'Negative']

        troponin_diff = positive_group['Troponin'].mean() - negative_group['Troponin'].mean()
        ckmb_diff = positive_group['CK-MB'].mean() - negative_group['CK-MB'].mean()

        print(f"   2. Różnice w biomarkerach (Pozytywne vs Negatywne):")
        print(f"      Troponina: +{troponin_diff:.3f} ng/mL ({troponin_diff/negative_group['Troponin'].mean()*100:+.1f}%)")
        print(f"      CK-MB: +{ckmb_diff:.3f} ng/mL ({ckmb_diff/negative_group['CK-MB'].mean()*100:+.1f}%)")

        # 3. Różnice płciowe
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
        """Interpretuje wartość AUC"""
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
        """Wyświetla rekomendacje dla przyszłych badań"""

        print(f"\n📋 REKOMENDACJE DLA PRZYSZŁYCH BADAŃ:")

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
        """Wyświetla ograniczenia badania"""

        print(f"\n⚠️  OGRANICZENIA BADANIA:")

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

    def create_report_summary(self, results, df, output_file="raport_podsumowanie.txt"):
        """Tworzy tekstowe podsumowanie do raportu"""

        print(f"\n📄 Tworzenie podsumowania raportu: {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RAPORT Z ANALIZY CZYNNIKÓW DIAGNOSTYCZNYCH ZAWAŁU SERCA\n")
            f.write("="*80 + "\n\n")

            # Sekcja 1: Wstęp
            f.write("1. WSTĘP\n")
            f.write("-"*20 + "\n\n")
            f.write(f"Projekt: {self.project_name}\n")
            f.write(f"Data: {self.date}\n")
            f.write(f"Liczba pacjentów: {len(df)}\n")
            f.write(f"Liczba zmiennych: {len(df.columns)}\n\n")

            f.write("Cel badania: Identyfikacja najważniejszych czynników diagnostycznych zawału serca\n")
            f.write("poprzez analizę związków między parametrami medycznymi a wystąpieniem zawału.\n\n")

            # Sekcja 2: Hipotezy
            f.write("2. HIPOTEZY BADAWCZE\n")
            f.write("-"*30 + "\n\n")

            hypotheses = [
                "H1: Wiek pacjenta jest głównym predyktorem poziomu troponiny",
                "H2: Płeć determinuje poziom ciśnienia skurczowego",
                "H3: Zawał serca można przewidzieć na podstawie biomarkerów"
            ]

            for i, hyp in enumerate(hypotheses, 1):
                f.write(f"Hipoteza {i}: {hyp}\n")
            f.write("\n")

            # Sekcja 3: Wyniki
            f.write("3. WYNIKI\n")
            f.write("-"*15 + "\n\n")

            confirmed_count = 0
            for hyp_id in ['h1', 'h2', 'h3']:
                if hyp_id in results:
                    result = results[hyp_id]
                    conclusion = result.get('hypothesis_conclusion', 'NIEZNANY')
                    if 'POTWIERDZONA' in conclusion:
                        confirmed_count += 1

                    f.write(f"Hipoteza {hyp_id.upper()}: {conclusion}\n")
                    f.write(f"Uzasadnienie: {result.get('hypothesis_explanation', 'Brak')}\n\n")

            # Sekcja 4: Wnioski
            f.write("4. WNIOSKI\n")
            f.write("-"*15 + "\n\n")

            success_rate = confirmed_count / 3 * 100
            f.write(f"• Potwierdzone hipotezy: {confirmed_count}/3 ({success_rate:.1f}%)\n")

            # Najsilniejszy predyktor
            numeric_vars = ['Age', 'Gender', 'Heart rate', 'Systolic blood pressure',
                            'Blood sugar', 'CK-MB', 'Troponin', 'Result_Binary']
            corr_matrix = df[numeric_vars].corr()
            result_corrs = corr_matrix['Result_Binary'].abs().drop('Result_Binary').sort_values(ascending=False)
            strongest_predictor = result_corrs.index[0]

            f.write(f"• Najsilniejszy predyktor zawału: {strongest_predictor}\n")
            f.write(f"• Średnia troponina u pacjentów z zawałem jest znacząco wyższa\n")
            f.write(f"• Modele predykcyjne osiągają zadowalającą dokładność\n\n")

            # Sekcja 5: Rekomendacje
            f.write("5. REKOMENDACJE\n")
            f.write("-"*20 + "\n\n")
            f.write("• Troponina powinna być priorytetowym biomarkerem w diagnostyce\n")
            f.write("• Kombinacja biomarkerów zwiększa dokładność diagnozy\n")
            f.write("• Konieczne są dalsze badania z większą próbą\n")
            f.write("• Należy uwzględnić dodatkowe czynniki kliniczne\n\n")

        print(f"✅ Podsumowanie zapisane do pliku: {output_file}")

    def generate_methodology_section(self):
        """Generuje sekcję metodologiczną dla raportu"""

        methodology = """
METODOLOGIA BADANIA

1. PRZYGOTOWANIE DANYCH:
   • Wczytanie datasetu medycznego (1319 pacjentów, 9 zmiennych)
   • Sprawdzenie braków danych i kompletności
   • Kodowanie zmiennej wynikowej (Result: Negative=0, Positive=1)

2. ANALIZA STATYSTYK OPISOWYCH (Punkty 1-6):
   • Statystyki podstawowe dla zmiennych ilościowych (średnia, mediana, odchylenie std.)
   • Tabele liczności dla zmiennych jakościowych
   • Tabele wielodzielcze (crosstab) z testami chi-kwadrat
   • Histogramy skategoryzowane według płci i wyniku
   • Wykresy średnich w grupach z analizą interakcji
   • Macierz korelacji Pearsona z wizualizacją heatmap

3. ZAAWANSOWANA ANALIZA (Punkty 7-10):
   • Test F dla oceny ważności zmiennych ilościowych
   • Test chi-kwadrat dla zmiennych jakościowych
   • Wykresy ramka-wąsy z wykryciem wartości odstających (metoda IQR)
   • Testy normalności (Shapiro-Wilk, D'Agostino-Pearson)
   • Wykresy rozrzutu dla skorelowanych par zmiennych

4. TESTOWANIE HIPOTEZ:
   • H1: Analiza regresyjna (Age, Gender, Heart rate → Troponin)
   • H2: Analiza regresyjna (Gender, Age, Blood sugar → Systolic BP)
   • H3: Analiza klasyfikacyjna (Troponin, CK-MB, Heart rate → Result)

5. MODELOWANIE:
   • Regresja liniowa (hipotezy 1-2): ocena R², RMSE, istotności współczynników
   • Regresja logistyczna (hipoteza 3): ocena accuracy, AUC, odds ratios
   • Podział danych: 70% trening, 30% test
   • Standaryzacja zmiennych dla modeli klasyfikacyjnych

6. WIZUALIZACJA:
   • Wykresy rozkładów podstawowych
   • Analizy związków między zmiennymi
   • Wykresy specyficzne dla każdej hipotezy
   • Dashboard podsumowujący wyniki
        """

        return methodology.strip()

    def create_tables_for_report(self, df, results):
        """Tworzy tabele do wstawienia w raporcie"""

        print("📊 Generowanie tabel do raportu...")

        tables = {}

        # Tabela 1: Statystyki opisowe
        quantitative_vars = ['Age', 'Heart rate', 'Systolic blood pressure',
                             'Blood sugar', 'CK-MB', 'Troponin']

        desc_stats = df[quantitative_vars].describe().round(3)
        tables['descriptive_stats'] = desc_stats

        # Tabela 2: Tabela kontyngencji
        contingency = pd.crosstab(df['Gender'], df['Result'], margins=True)
        tables['contingency'] = contingency

        # Tabela 3: Korelacje z wynikiem
        correlations = df[quantitative_vars + ['Result_Binary']].corr()['Result_Binary'].drop('Result_Binary')
        corr_df = pd.DataFrame({
            'Zmienna': correlations.index,
            'Korelacja': correlations.values.round(3),
            'Interpretacja': [self._interpret_correlation(abs(r)) for r in correlations.values]
        })
        tables['correlations'] = corr_df

        # Tabela 4: Wyniki hipotez
        if results:
            hyp_results = []
            for hyp_id in ['h1', 'h2', 'h3']:
                if hyp_id in results:
                    result = results[hyp_id]
                    if result['type'] == 'regression':
                        metric = f"R² = {result.get('r2_test', 0):.3f}"
                    else:
                        metric = f"AUC = {result.get('auc', 0):.3f}"

                    hyp_results.append({
                        'Hipoteza': hyp_id.upper(),
                        'Status': result.get('hypothesis_conclusion', 'NIEZNANY'),
                        'Metryka': metric
                    })

            tables['hypothesis_results'] = pd.DataFrame(hyp_results)

        return tables

    def _interpret_correlation(self, abs_corr):
        """Interpretuje siłę korelacji"""
        if abs_corr < 0.1:
            return "bardzo słaba"
        elif abs_corr < 0.3:
            return "słaba"
        elif abs_corr < 0.5:
            return "umiarkowana"
        elif abs_corr < 0.7:
            return "silna"
        else:
            return "bardzo silna"

    def print_citation_info(self):
        """Wyświetla informacje o cytowaniu"""

        print(f"\n📚 INFORMACJE O CYTOWANIU:")
        print("-" * 40)
        print(f"Tytuł: {self.project_name}")
        print(f"Autor: {self.author}")
        print(f"Data: {self.date}")
        print(f"Metodologia: Analiza statystyczna z wykorzystaniem Python/pandas/scipy")
        print(f"Dataset: Medical Dataset - Heart Attack Prediction")

    def get_project_info(self):
        """Zwraca informacje o projekcie w formacie słownika"""

        return {
            'project_name': self.project_name,
            'author': self.author,
            'date': self.date,
            'description': "Analiza statystyczna czynników diagnostycznych zawału serca",
            'methodology': self.generate_methodology_section(),
            'tools_used': ['Python', 'pandas', 'numpy', 'scipy', 'scikit-learn', 'matplotlib', 'seaborn']
        }