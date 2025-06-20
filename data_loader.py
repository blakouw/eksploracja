"""
MODUŁ WCZYTYWANIA I PRZYGOTOWANIA DANYCH
Klasa odpowiedzialna za wczytanie datasetu i jego wstępne przygotowanie
"""

import pandas as pd
import numpy as np

class DataLoader:
    """Klasa do wczytywania i przygotowywania danych medycznych"""

    def __init__(self):
        self.df = None
        self.df_analysis = None

    def load_and_prepare_data(self, filepath):
        """
        Wczytuje i przygotowuje dane do analizy

        Args:
            filepath (str): Ścieżka do pliku CSV

        Returns:
            tuple: (df_original, df_analysis)
        """

        print(f"📂 Wczytywanie danych z pliku: {filepath}")

        try:
            # Wczytanie danych
            self.df = pd.read_csv(filepath)
            print(f"✅ Pomyślnie wczytano {self.df.shape[0]} wierszy i {self.df.shape[1]} kolumn")

            # Przygotowanie danych do analizy
            self.df_analysis = self._prepare_analysis_data()

            # Podstawowe informacje
            self._display_basic_info()

            # Wyświetlenie hipotez
            self._display_hypotheses()

            return self.df, self.df_analysis

        except FileNotFoundError:
            print(f"❌ Błąd: Nie znaleziono pliku {filepath}")
            raise
        except Exception as e:
            print(f"❌ Błąd podczas wczytywania danych: {e}")
            raise

    def _prepare_analysis_data(self):
        """Przygotowuje dane do analizy (kodowanie zmiennych itp.)"""

        df_analysis = self.df.copy()

        # Kodowanie zmiennej Result na binarną
        df_analysis['Result_Binary'] = df_analysis['Result'].map({
            'Negative': 0,
            'Positive': 1
        })

        print("🔧 Przygotowano zmienne do analizy:")
        print("   - Result_Binary: 0=Negative, 1=Positive")

        return df_analysis

    def _display_basic_info(self):
        """Wyświetla podstawowe informacje o danych"""

        print("\n📋 PODSTAWOWE INFORMACJE O DANYCH:")
        print("-" * 50)
        print(f"Wymiary datasetu: {self.df.shape[0]} wierszy, {self.df.shape[1]} kolumn")

        print("\n📝 Opis zmiennych:")
        variable_descriptions = {
            'Age': 'Wiek pacjenta (lata)',
            'Gender': 'Płeć (1=mężczyzna, 0=kobieta)',
            'Heart rate': 'Tętno (uderzenia/min)',
            'Systolic blood pressure': 'Ciśnienie skurczowe (mmHg)',
            'Diastolic blood pressure': 'Ciśnienie rozkurczowe (mmHg)',
            'Blood sugar': 'Poziom glukozy we krwi (mg/dL)',
            'CK-MB': 'Enzym sercowy (ng/mL)',
            'Troponin': 'Białko biomarker uszkodzenia mięśnia sercowego (ng/mL)',
            'Result': 'Wynik (Positive=zawał, Negative=brak zawału)'
        }

        for col, desc in variable_descriptions.items():
            if col in self.df.columns:
                print(f"   • {col}: {desc}")

        # Sprawdzenie braków danych
        missing_total = self.df.isnull().sum().sum()
        print(f"\n🔍 Braki danych: {missing_total}")

        if missing_total > 0:
            print("Szczegóły braków danych:")
            missing_details = self.df.isnull().sum()
            for col, missing in missing_details.items():
                if missing > 0:
                    print(f"   • {col}: {missing} ({missing/len(self.df)*100:.1f}%)")
        else:
            print("✅ Dataset kompletny - brak braków danych")

        # Rozkład zmiennej docelowej
        result_counts = self.df['Result'].value_counts()
        print(f"\n🎯 Rozkład zmiennej docelowej (Result):")
        print(f"   • Positive (zawał): {result_counts.get('Positive', 0)} ({result_counts.get('Positive', 0)/len(self.df)*100:.1f}%)")
        print(f"   • Negative (brak zawału): {result_counts.get('Negative', 0)} ({result_counts.get('Negative', 0)/len(self.df)*100:.1f}%)")

    def _display_hypotheses(self):
        """Wyświetla sformułowane hipotezy badawcze"""

        print("\n🎯 SFORMUŁOWANE HIPOTEZY BADAWCZE:")
        print("-" * 50)

        hypotheses = [
            {
                'number': 1,
                'title': 'Wiek pacjenta jest głównym predyktorem poziomu troponiny',
                'dependent': 'Troponin (ilościowa)',
                'independent': 'Age, Gender, Heart rate'
            },
            {
                'number': 2,
                'title': 'Płeć determinuje poziom ciśnienia skurczowego',
                'dependent': 'Systolic blood pressure (ilościowa)',
                'independent': 'Gender, Age, Blood sugar'
            },
            {
                'number': 3,
                'title': 'Zawał serca można przewidzieć na podstawie biomarkerów',
                'dependent': 'Result (jakościowa)',
                'independent': 'Troponin, CK-MB, Heart rate'
            }
        ]

        for hyp in hypotheses:
            print(f"\n📌 HIPOTEZA {hyp['number']}: {hyp['title']}")
            print(f"   • Zmienna zależna: {hyp['dependent']}")
            print(f"   • Zmienne objaśniające: {hyp['independent']}")

    def get_variable_info(self):
        """Zwraca informacje o zmiennych w formacie słownika"""

        if self.df_analysis is None:
            raise ValueError("Dane nie zostały jeszcze wczytane")

        # Podział na zmienne ilościowe i jakościowe
        quantitative_vars = ['Age', 'Heart rate', 'Systolic blood pressure',
                             'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']

        qualitative_vars = ['Gender', 'Result']

        # Zmienne dla każdej hipotezy
        hypothesis_vars = {
            'h1': {
                'dependent': 'Troponin',
                'independent': ['Age', 'Gender', 'Heart rate'],
                'type': 'regression'
            },
            'h2': {
                'dependent': 'Systolic blood pressure',
                'independent': ['Gender', 'Age', 'Blood sugar'],
                'type': 'regression'
            },
            'h3': {
                'dependent': 'Result_Binary',
                'independent': ['Troponin', 'CK-MB', 'Heart rate'],
                'type': 'classification'
            }
        }

        return {
            'quantitative': quantitative_vars,
            'qualitative': qualitative_vars,
            'hypotheses': hypothesis_vars,
            'all_numeric': quantitative_vars + ['Gender', 'Result_Binary']
        }

    def get_data_summary(self):
        """Zwraca podsumowanie danych w formacie słownika"""

        if self.df_analysis is None:
            raise ValueError("Dane nie zostały jeszcze wczytane")

        return {
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().sum(),
            'result_distribution': self.df['Result'].value_counts().to_dict(),
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict()
        }