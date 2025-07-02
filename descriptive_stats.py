
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

class DescriptiveStats:

    def __init__(self, df_analysis):
        self.df = df_analysis
        self.quantitative_vars = ['Age', 'Heart rate', 'Systolic blood pressure',
                                  'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']
        self.qualitative_vars = ['Gender', 'Result']
        self.all_numeric = self.quantitative_vars + ['Gender', 'Result_Binary']

    def run_all_analyses(self):

        print("\n" + "=" * 70)
        print("CZĘŚĆ II: CZYSZCZENIE I ANALIZA DANYCH - STATYSTYKI OPISOWE")
        print("=" * 70)

        # Punkt 1: Statystyki opisowe dla zmiennych ilościowych
        self.calculate_quantitative_statistics()

        # Punkt 2: Tabele liczności dla zmiennych jakościowych
        self.calculate_qualitative_frequencies()

        # Punkt 3: Tabela wielodzielcza
        self.create_contingency_table()

        # Punkt 4: Histogramy skategoryzowane
        self.create_categorized_histograms()

        # Punkt 5: Wykresy średnich w grupach
        self.create_group_means_plots()

        # Punkt 6: Macierz korelacji
        self.calculate_correlation_matrix()

    def calculate_quantitative_statistics(self):

        print("\n1. STATYSTYKI OPISOWE DLA ZMIENNYCH ILOŚCIOWYCH")
        print("-" * 60)

        self.quant_stats = {}

        for var in self.quantitative_vars:
            print(f"\n{var.upper()}:")
            data = self.df[var]

            stats_dict = {
                'Liczba przypadków': len(data),
                'Średnia': data.mean(),
                'Mediana': data.median(),
                'Moda': data.mode().iloc[0] if len(data.mode()) > 0 else np.nan,
                'Minimum': data.min(),
                'Maksimum': data.max(),
                'Odchylenie std': data.std(),
                'Wariancja': data.var()
            }

            self.quant_stats[var] = stats_dict

            for key, value in stats_dict.items():
                if isinstance(value, (int, float)) and key != 'Liczba przypadków':
                    print(f"   {key}: {value:.3f}")
                else:
                    print(f"   {key}: {value}")

        self._create_summary_table()

    def _create_summary_table(self):

        print(f"\nTABELA PODSUMOWUJĄCA STATYSTYKI:")

        summary_df = pd.DataFrame()
        for var, stats in self.quant_stats.items():
            summary_df[var] = [
                stats['Liczba przypadków'],
                round(stats['Średnia'], 3),
                round(stats['Mediana'], 3),
                round(stats['Minimum'], 3),
                round(stats['Maksimum'], 3),
                round(stats['Odchylenie std'], 3),
                round(stats['Wariancja'], 3)
            ]

        summary_df.index = ['N', 'Średnia', 'Mediana', 'Min', 'Max', 'Std', 'Wariancja']
        print(summary_df.to_string())

    def calculate_qualitative_frequencies(self):

        print("\n\n2. TABELE LICZNOŚCI DLA ZMIENNYCH JAKOŚCIOWYCH")
        print("-" * 60)

        self.qual_stats = {}

        for var in self.qualitative_vars:
            print(f"\n {var.upper()}:")

            counts = self.df[var].value_counts()
            percentages = self.df[var].value_counts(normalize=True) * 100

            self.qual_stats[var] = {
                'counts': counts,
                'percentages': percentages
            }

            for category in counts.index:
                print(f"   {category}: {counts[category]} ({percentages[category]:.1f}%)")

            print(f"   Łącznie: {counts.sum()} przypadków")

    def create_contingency_table(self):

        print("\n\n3. TABELA WIELODZIELCZA")
        print("-" * 40)

        self.contingency_table = pd.crosstab(
            self.df['Gender'],
            self.df['Result'],
            margins=True
        )

        print("📊 GENDER vs RESULT (liczności):")
        print(self.contingency_table)

        self.contingency_pct = pd.crosstab(
            self.df['Gender'],
            self.df['Result'],
            normalize='index'
        ) * 100

        print("\nGENDER vs RESULT (procenty względem płci):")
        print(self.contingency_pct.round(1))

        contingency_pct_result = pd.crosstab(
            self.df['Gender'],
            self.df['Result'],
            normalize='columns'
        ) * 100

        print("\nGENDER vs RESULT (procenty względem wyniku):")
        print(contingency_pct_result.round(1))

        chi2_stat, p_val, dof, expected = chi2_contingency(
            self.contingency_table.iloc[:-1, :-1]
        )

        print(f"\nTEST CHI-KWADRAT:")
        print(f"   Chi² = {chi2_stat:.3f}")
        print(f"   p-value = {p_val:.6f}")
        print(f"   Stopnie swobody = {dof}")
        print(f"   Wynik: {'Istotna zależność' if p_val < 0.05 else 'Brak istotnej zależności'} (α=0.05)")

    def create_categorized_histograms(self):

        print("\n\n4. HISTOGRAMY SKATEGORYZOWANE")
        print("-" * 45)

        print(f"\nDEBUG - Analiza zmiennej Result:")
        print(f"  Unikalne wartości: {self.df['Result'].unique()}")
        print(f"  Rozkład: {self.df['Result'].value_counts()}")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Histogramy skategoryzowane według płci i wyniku', fontsize=16, fontweight='bold')

        vars_for_gender = ['Troponin', 'Systolic blood pressure', 'Age']

        for i, var in enumerate(vars_for_gender):
            # Dane dla każdej płci
            women_data = self.df[self.df['Gender'] == 0][var].dropna()
            men_data = self.df[self.df['Gender'] == 1][var].dropna()

            # Sprawdź czy mamy wystarczająco danych
            if len(women_data) > 5:
                axes[0, i].hist(women_data, alpha=0.6, label=f'Kobiety (n={len(women_data)})',
                                bins=20, density=True, color='pink')

            if len(men_data) > 5:
                axes[0, i].hist(men_data, alpha=0.6, label=f'Mężczyźni (n={len(men_data)})',
                                bins=20, density=True, color='lightblue')

            axes[0, i].set_title(f'{var} według płci')
            axes[0, i].set_xlabel(var)
            axes[0, i].set_ylabel('Gęstość')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)

        vars_for_result = ['Troponin', 'CK-MB', 'Heart rate']

        unique_results = self.df['Result'].unique()
        print(f"\nDEBUG - Rysowanie histogramów dla wyników: {unique_results}")

        for i, var in enumerate(vars_for_result):
            any_plotted = False

            for j, result_value in enumerate(unique_results):
                data = self.df[self.df['Result'] == result_value][var].dropna()

                if len(data) > 5:
                    if 'neg' in str(result_value).lower():
                        color = 'lightgreen'
                        label = 'Negative'
                    elif 'pos' in str(result_value).lower():
                        color = 'lightcoral'
                        label = 'Positive'
                    else:
                        color = f'C{j}'
                        label = str(result_value)

                    label_with_count = f'{label} (n={len(data)})'

                    axes[1, i].hist(data, alpha=0.6, label=label_with_count,
                                    bins='auto', color=color, edgecolor='black')
                    axes[1, i].set_yscale('log')
                    any_plotted = True

                    print(f"  DEBUG - {var} dla {result_value}: {len(data)} obserwacji")
                else:
                    print(f"  DEBUG - {var} dla {result_value}: tylko {len(data)} obserwacji (za mało)")

            if not any_plotted:
                axes[1, i].text(0.5, 0.5, 'Brak wystarczających danych\ndla tej zmiennej',
                                ha='center', va='center', transform=axes[1, i].transAxes,
                                fontsize=12, color='red')

            axes[1, i].set_title(f'{var} według wyniku')
            axes[1, i].set_xlabel(var)
            axes[1, i].set_ylabel('Gęstość')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("✅ Histogramy skategoryzowane zostały wygenerowane")

    def create_group_means_plots(self):

        print("\n5. WYKRES ŚREDNICH W GRUPACH")
        print("-" * 40)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Wykresy średnich w grupach i interakcji', fontsize=16, fontweight='bold')

        gender_means = self.df.groupby('Gender')[['Troponin', 'Systolic blood pressure', 'Age']].mean()
        gender_means.T.plot(kind='bar', ax=axes[0,0], color=['pink', 'lightblue'])
        axes[0,0].set_title('Średnie zmiennych według płci')
        axes[0,0].set_xlabel('Zmienne')
        axes[0,0].set_ylabel('Średnia wartość')
        axes[0,0].legend(['Kobiety (0)', 'Mężczyźni (1)'])
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)

        print("Średnie według płci:")
        print(gender_means.round(2))

        result_means = self.df.groupby('Result')[['Troponin', 'CK-MB', 'Heart rate']].mean()
        result_means.T.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Średnie zmiennych według wyniku')
        axes[0,1].set_xlabel('Zmienne')
        axes[0,1].set_ylabel('Średnia wartość')

        legend_labels = [str(val).capitalize() for val in result_means.index]
        axes[0,1].legend(legend_labels)
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)

        print("\nŚrednie według wyniku:")
        print(result_means.round(2))

        interaction_data = []
        for gender in [0, 1]:
            for result in self.df['Result'].unique():
                subset = self.df[(self.df['Gender'] == gender) & (self.df['Result'] == result)]
                if len(subset) > 0:
                    mean_troponin = subset['Troponin'].mean()
                    interaction_data.append({
                        'Gender': 'Kobiety' if gender == 0 else 'Mężczyźni',
                        'Result': str(result).capitalize(),
                        'Mean_Troponin': mean_troponin
                    })

        gender_groups = ['Kobiety', 'Mężczyźni']
        result_values = [str(r).capitalize() for r in self.df['Result'].unique()]

        for gender in gender_groups:
            gender_data = [d for d in interaction_data if d['Gender'] == gender]
            if gender_data:
                results = [d['Result'] for d in gender_data]
                means = [d['Mean_Troponin'] for d in gender_data]
                axes[1,0].plot(results, means, marker='o', linewidth=2, label=gender)

        axes[1,0].set_title('Interakcja: Płeć × Wynik → Troponina')
        axes[1,0].set_xlabel('Wynik')
        axes[1,0].set_ylabel('Średnia Troponina')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # 5d. Boxplot porównawczy
        box_data = [
            self.df[self.df['Gender'] == 0]['Systolic blood pressure'].dropna(),
            self.df[self.df['Gender'] == 1]['Systolic blood pressure'].dropna()
        ]

        bp = axes[1,1].boxplot(box_data, labels=['Kobiety', 'Mężczyźni'], patch_artist=True)
        bp['boxes'][0].set_facecolor('pink')
        bp['boxes'][1].set_facecolor('lightblue')
        axes[1,1].set_title('Rozkład ciśnienia skurczowego według płci')
        axes[1,1].set_ylabel('Ciśnienie skurczowe (mmHg)')
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("Wykresy średnich w grupach zostały wygenerowane")

    def calculate_correlation_matrix(self):
        """Punkt 6: Macierz korelacji"""

        print("\n6. MACIERZ KORELACJI")
        print("-" * 30)

        # Oblicz macierz korelacji dla wszystkich zmiennych numerycznych
        self.correlation_matrix = self.df[self.all_numeric].corr()

        print("mACIERZ KORELCJI (wszystkie zmienne numeryczne):")
        print(self.correlation_matrix.round(3))

        self._find_strongest_correlations()

        self._plot_correlation_matrix()

        print("Analiza korcji została zakończona")

    def _find_strongest_correlations(self):

        print(f"\nl NAJSILNIEJSZE KORELACJE:")

        correlations = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                var1 = self.correlation_matrix.columns[i]
                var2 = self.correlation_matrix.columns[j]
                corr_value = self.correlation_matrix.iloc[i, j]

                if not pd.isna(corr_value):
                    correlations.append((var1, var2, corr_value, abs(corr_value)))

        if not correlations:
            print("Brk prawidłowych korelacji do wyświetlenia")
            return

        correlations.sort(key=lambda x: x[3], reverse=True)

        print("Top 5 najsilniejszych korelacji:")
        for i, (var1, var2, corr, abs_corr) in enumerate(correlations[:5], 1):
            strength = self._interpret_correlation_strength(abs_corr)
            direction = "dodatnia" if corr > 0 else "ujemna"
            print(f"   {i}. {var1} ↔ {var2}: r = {corr:.3f} ({direction}, {strength})")

    def _interpret_correlation_strength(self, abs_corr):
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

    def _plot_correlation_matrix(self):

        plt.figure(figsize=(12, 10))

        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))

        sns.heatmap(
            self.correlation_matrix,
            mask=mask,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.3f',
            cbar_kws={"shrink": .8},
            linewidths=0.5
        )

        plt.title('Macierz korelacji wszystkich zmiennych numerycznych', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def get_statistics_summary(self):

        return {
            'quantitative_stats': self.quant_stats,
            'qualitative_stats': self.qual_stats,
            'contingency_table': self.contingency_table,
            'correlation_matrix': self.correlation_matrix
        }