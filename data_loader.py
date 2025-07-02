import pandas as pd
import numpy as np

class DataLoader:

    def __init__(self):
        self.df = None
        self.df_analysis = None

    def load_and_prepare_data(self, filepath):

        print(f"Wczytywanie danych z pliku: {filepath}")

        try:
            self.df = pd.read_csv(filepath)
            print(f"✅ Pomyślnie wczytano {self.df.shape[0]} wierszy i {self.df.shape[1]} kolumn")

            self.df_analysis = self._prepare_analysis_data()

            self._display_basic_info()

            self._display_hypotheses()

            return self.df, self.df_analysis

        except FileNotFoundError:
            print(f"❌ Błąd: Nie znaleziono pliku {filepath}")
            raise
        except Exception as e:
            print(f"❌ Błąd podczas wczytywania danych: {e}")
            raise

    def _prepare_analysis_data(self):

        df_analysis = self.df.copy()

        print("🔧 Przygotowanie danych do analizy...")
        print(f"   Początkowa liczba wierszy: {len(df_analysis)}")

        initial_rows = len(df_analysis)
        df_analysis = df_analysis.dropna(subset=['Result'])
        after_result_clean = len(df_analysis)

        if initial_rows > after_result_clean:
            print(f"🧹 Usunięto {initial_rows - after_result_clean} wierszy z brakującymi Result")

        print(f"   Unikalne wartości w Result przed kodowaniem: {df_analysis['Result'].unique()}")

        df_analysis['Result_Binary'] = df_analysis['Result'].str.lower().map({
            'negative': 0,
            'positive': 1
        })

        null_binary = df_analysis['Result_Binary'].isnull().sum()
        if null_binary > 0:
            print(f"PROBLEM: {null_binary} wartości Result nie zostało zakodowanych!")
            print("Unikalne wartości w Result po .lower():", df_analysis['Result'].str.lower().unique())
            df_analysis = df_analysis.dropna(subset=['Result_Binary'])
            print(f"🧹 Usunięto {null_binary} wierszy z problemami kodowania")
        else:
            print(f"Kodw Result_Binary zakończone sukcesem")

        numeric_cols = ['Age', 'Heart rate', 'Systolic blood pressure',
                        'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']

        for col in numeric_cols:
            if col in df_analysis.columns:
                initial_col_size = len(df_analysis)

                df_analysis[col] = df_analysis[col].replace([np.inf, -np.inf], np.nan)

                if df_analysis[col].dtype == 'object':
                    print(f"⚠️  {col} ma typ object - próba konwersji na numeric")
                    df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')

                df_analysis = df_analysis.dropna(subset=[col])

                final_col_size = len(df_analysis)
                if initial_col_size > final_col_size:
                    print(f"🧹 {col}: usunięto {initial_col_size - final_col_size} wierszy z NaN")

        final_rows = len(df_analysis)
        print(f"✅ Finalna liczba wierszy: {final_rows}")

        if df_analysis['Result_Binary'].isnull().sum() > 0:
            raise ValueError("❌ Result_Binary nadal zawiera NaN po czyszczeniu!")

        result_counts = df_analysis['Result_Binary'].value_counts()
        print(f"📊 Rozkład Result_Binary: {result_counts.to_dict()}")

        print("🔧 Przygotowano zmienne do analizy:")
        print("   - Result_Binary: 0=Negative, 1=Positive")
        print("   - Wszystkie zmienne numeryczne oczyszczone z NaN/inf")

        return df_analysis

    def _display_basic_info(self):

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

        result_counts = self.df['Result'].value_counts()
        print(f"\n🎯 Rozkład zmiennej docelowej (Result):")

        positive_count = 0
        negative_count = 0

        for value, count in result_counts.items():
            if str(value).lower() == 'positive':
                positive_count = count
            elif str(value).lower() == 'negative':
                negative_count = count
            print(f"   • {value}: {count} ({count/len(self.df)*100:.1f}%)")

        print(f"\n📊 Podsumowanie (po normalizacji nazw):")
        print(f"   • Positive (zawał): {positive_count} ({positive_count/len(self.df)*100:.1f}%)")
        print(f"   • Negative (brak zawału): {negative_count} ({negative_count/len(self.df)*100:.1f}%)")

    def _display_hypotheses(self):

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
            print(f"\nHIPOTEZA {hyp['number']}: {hyp['title']}")
            print(f"   • Zmienna zależna: {hyp['dependent']}")
            print(f"   • Zmienne objaśniające: {hyp['independent']}")

    def get_variable_info(self):

        if self.df_analysis is None:
            raise ValueError("Dane nie zostały jeszcze wczytane")

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

        if self.df_analysis is None:
            raise ValueError("Dane nie zostały jeszcze wczytane")

        return {
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().sum(),
            'result_distribution': self.df['Result'].value_counts().to_dict(),
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict()
        }