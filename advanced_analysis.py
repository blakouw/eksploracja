import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, chi2_contingency
from sklearn.feature_selection import f_regression, SelectKBest

class AdvancedAnalysis:

    def __init__(self, df_analysis):
        self.df = df_analysis
        self.quantitative_vars = ['Age', 'Heart rate', 'Systolic blood pressure',
                                  'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']
        self.all_numeric = self.quantitative_vars + ['Gender', 'Result_Binary']

    def run_all_analyses(self):

        print("\n" + "=" * 70)
        print("ZAAWANSOWANA ANALIZA DANYCH")
        print("=" * 70)

        # Punkt 7: Test Chi² i ważność zmiennych
        self.perform_chi2_and_importance_tests()

        # Punkt 8: Wykresy ramka-wąsy
        self.create_boxplots()

        # Punkt 9: Test normalności i wartości odstające
        self.test_normality_and_outliers()

        # Punkt 10: Wykresy rozrzutu
        self.create_scatter_plots()

    def perform_chi2_and_importance_tests(self):
        """Punkt 7: Test Chi² i ważność zmiennych"""

        print("\n7. TEST CHI² I WAŻNOŚĆ ZMIENNYCH")
        print("-" * 50)

        # 7a. Test F dla zmiennych ilościowych vs Result_Binary
        print("📊 WAŻNOŚĆ ZMIENNYCH ILOŚCIOWYCH (F-test):")

        X_quant = self.df[self.quantitative_vars].copy()
        y_binary = self.df['Result_Binary'].copy()

        # Sprawdź czy dane są czyste
        print(f"   Sprawdzanie danych:")
        print(f"   X shape: {X_quant.shape}")
        print(f"   y shape: {y_binary.shape}")
        print(f"   NaN w X: {X_quant.isnull().sum().sum()}")
        print(f"   NaN w y: {y_binary.isnull().sum()}")

        # Usuń wiersze z NaN
        mask = ~(X_quant.isnull().any(axis=1) | y_binary.isnull())
        X_clean = X_quant[mask]
        y_clean = y_binary[mask]

        print(f"   Po czyszczeniu - X shape: {X_clean.shape}, y shape: {y_clean.shape}")

        if len(X_clean) == 0:
            print("❌ Brak danych po czyszczeniu!")
            return

        # Sprawdź czy y ma wariancję
        if y_clean.nunique() < 2:
            print("❌ Zmienna docelowa nie ma wariancji!")
            return

        # Oblicz F-scores
        try:
            f_scores, p_values = f_regression(X_clean, y_clean)

            # Stwórz DataFrame z wynikami
            self.importance_df = pd.DataFrame({
                'Zmienna': X_clean.columns,
                'F-score': f_scores,
                'p-value': p_values,
                'Istotność': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                              for p in p_values]
            }).sort_values('F-score', ascending=False)

            print(self.importance_df.round(4))
            print(f"\nLegenda istotności: *** p<0.001, ** p<0.01, * p<0.05, ns = nie istotne")

        except Exception as e:
            print(f"Błąd w F-test: {e}")
            self.importance_df = pd.DataFrame({
                'Zmienna': self.quantitative_vars,
                'F-score': [0] * len(self.quantitative_vars),
                'p-value': [1] * len(self.quantitative_vars),
                'Istotność': ['ns'] * len(self.quantitative_vars)
            })

        print(f"\nTEST CHI² (Gender vs Result):")

        try:
            contingency = pd.crosstab(self.df['Gender'], self.df['Result'])
            chi2_stat, p_val, dof, expected = chi2_contingency(contingency)

            print(f"   Chi² = {chi2_stat:.3f}")
            print(f"   p-value = {p_val:.6f}")
            print(f"   Stopnie swobody = {dof}")
            print(f"   Wynik: {'Istotna zależność' if p_val < 0.05 else 'Brak istotnej zależności'} (α=0.05)")

        except Exception as e:
            print(f"Błąd w teście Chi²: {e}")

        self._plot_variable_importance()

    def _plot_variable_importance(self):

        plt.figure(figsize=(12, 8))

        colors = ['red' if p < 0.05 else 'lightblue' for p in self.importance_df['p-value']]

        bars = plt.barh(self.importance_df['Zmienna'], self.importance_df['F-score'], color=colors)

        for i, (bar, f_score, p_val) in enumerate(zip(bars, self.importance_df['F-score'],
                                                      self.importance_df['p-value'])):
            significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            plt.text(bar.get_width() + max(self.importance_df['F-score']) * 0.01,
                     bar.get_y() + bar.get_height()/2,
                     f'{f_score:.2f}{significance}',
                     va='center', fontsize=10)

        plt.xlabel('F-score')
        plt.ylabel('Zmienne')
        plt.title('Ważność zmiennych w przewidywaniu zawału serca\n(czerwone = istotne statystycznie)',
                  fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')

        # Legenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Istotne statystycznie (p < 0.05)'),
            Patch(facecolor='lightblue', label='Nieistotne statystycznie (p ≥ 0.05)')
        ]
        plt.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.show()

        print("Wykres ważności zmiennych został wygenerowany")

    def create_boxplots(self):

        print("\n8. WYKRESY RAMKA-WĄSY")
        print("-" * 30)

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Wykresy ramka-wąsy dla wszystkich zmiennych ilościowych',
                     fontsize=16, fontweight='bold')

        for i, var in enumerate(self.quantitative_vars):
            row = i // 3
            col = i % 3

            box_data = self.df[var].dropna()
            axes[row, col].boxplot(box_data)
            axes[row, col].set_title(f'{var}')
            axes[row, col].set_ylabel(var)
            axes[row, col].grid(True, alpha=0.3)

            median = box_data.median()
            q1 = box_data.quantile(0.25)
            q3 = box_data.quantile(0.75)
            axes[row, col].text(0.02, 0.98, f'Med: {median:.1f}\nQ1: {q1:.1f}\nQ3: {q3:.1f}',
                                transform=axes[row, col].transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                fontsize=9)


        sns.boxplot(data=self.df, x='Gender', y='Troponin', ax=axes[2, 1])
        axes[2, 1].set_title('Troponina według płci')
        axes[2, 1].set_xticklabels(['Kobiety', 'Mężczyźni'])
        axes[2, 1].grid(True, alpha=0.3)

        sns.boxplot(data=self.df, x='Result', y='CK-MB', ax=axes[2, 2])
        axes[2, 2].set_title('CK-MB według wyniku')
        axes[2, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        self._create_additional_categorized_boxplots()

        print("Wykresy ramka-wąsy zostały wygenerowane")

    def _create_additional_categorized_boxplots(self):

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dodatkowe wykresy ramka-wąsy skategoryzowane', fontsize=16, fontweight='bold')

        plot_configs = [
            ('Gender', 'Age', 'Wiek według płci'),
            ('Result', 'Heart rate', 'Tętno według wyniku'),
            ('Gender', 'Blood sugar', 'Cukier według płci'),
            ('Result', 'Systolic blood pressure', 'Ciśnienie według wyniku')
        ]

        for i, (cat_var, quant_var, title) in enumerate(plot_configs):
            row = i // 2
            col = i % 2

            sns.boxplot(data=self.df, x=cat_var, y=quant_var, ax=axes[row, col])
            axes[row, col].set_title(title)
            axes[row, col].grid(True, alpha=0.3)

            if cat_var == 'Gender':
                axes[row, col].set_xticklabels(['Kobiety', 'Mężczyźni'])

        plt.tight_layout()
        plt.show()

    def test_normality_and_outliers(self):

        print("\n9. TEST NORMALNOŚCI I WARTOŚCI ODSTAJĄCE")
        print("-" * 50)

        self.normality_results = {}
        self.outliers_info = {}

        for var in self.quantitative_vars:
            print(f"\n📊 {var.upper()}:")

            data = self.df[var].dropna()

            if len(data) < 5000:
                statistic, p_value = shapiro(data)
                test_name = "Shapiro-Wilk"
            else:
                statistic, p_value = normaltest(data)
                test_name = "D'Agostino-Pearson"

            is_normal = p_value > 0.05

            print(f"   Test normalności ({test_name}):")
            print(f"   Statystyka: {statistic:.4f}")
            print(f"   p-value: {p_value:.6f}")
            print(f"   Rozkład: {'NORMALNY' if is_normal else 'NIE-NORMALNY'} (α=0.05)")

            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = data[(data < lower_bound) | (data > upper_bound)]
            outlier_percentage = len(outliers) / len(data) * 100

            print(f"   Wartości odstające (metoda IQR):")
            print(f"   Liczba: {len(outliers)} ({outlier_percentage:.1f}%)")
            print(f"   Zakres normalny: [{lower_bound:.2f}, {upper_bound:.2f}]")

            if len(outliers) > 0:
                print(f"   Zakres outliers: [{outliers.min():.2f}, {outliers.max():.2f}]")

            if is_normal:
                z_scores = np.abs(stats.zscore(data))
                z_outliers = data[z_scores > 3]  # |Z| > 3

                print(f"   Outliers metodą Z-score (|Z| > 3): {len(z_outliers)} ({len(z_outliers)/len(data)*100:.1f}%)")

            self.normality_results[var] = {
                'test_name': test_name,
                'statistic': statistic,
                'p_value': p_value,
                'is_normal': is_normal
            }

            self.outliers_info[var] = {
                'iqr_outliers': len(outliers),
                'iqr_percentage': outlier_percentage,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_values': outliers.tolist() if len(outliers) <= 10 else f"({len(outliers)} wartości)"
            }

        self._create_normality_summary()

        self._plot_distributions_with_outliers()

    def _create_normality_summary(self):

        print(f"\nPODSUMOWANIE TESTÓW NORMALNOŚCI:")

        summary_data = []
        for var, results in self.normality_results.items():
            outlier_count = self.outliers_info[var]['iqr_outliers']
            outlier_pct = self.outliers_info[var]['iqr_percentage']

            summary_data.append({
                'Zmienna': var,
                'Test': results['test_name'],
                'p-value': results['p_value'],
                'Normalny': 'Tak' if results['is_normal'] else 'Nie',
                'Outliers': f"{outlier_count} ({outlier_pct:.1f}%)"
            })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        normal_vars = sum(1 for r in self.normality_results.values() if r['is_normal'])
        total_vars = len(self.normality_results)

        print(f"\n📈 STATYSTYKI OGÓLNE:")
        print(f"   Zmienne o rozkładzie normalnym: {normal_vars}/{total_vars} ({normal_vars/total_vars*100:.1f}%)")

        total_outliers = sum(self.outliers_info[var]['iqr_outliers'] for var in self.quantitative_vars)
        total_observations = len(self.df) * len(self.quantitative_vars)
        print(f"   Łączna liczba outliers: {total_outliers} ({total_outliers/total_observations*100:.2f}% wszystkich obserwacji)")

    def _plot_distributions_with_outliers(self):

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Rozkłady zmiennych z zaznaczonymi wartościami odstającymi',
                     fontsize=16, fontweight='bold')

        for i, var in enumerate(self.quantitative_vars):
            row = i // 3
            col = i % 3

            data = self.df[var].dropna()

            axes[row, col].hist(data, bins=30, alpha=0.7, color='lightblue', density=True)

            outlier_info = self.outliers_info[var]
            lower_bound = outlier_info['lower_bound']
            upper_bound = outlier_info['upper_bound']

            axes[row, col].axvline(lower_bound, color='red', linestyle='--', alpha=0.7, label='Granica outliers')
            axes[row, col].axvline(upper_bound, color='red', linestyle='--', alpha=0.7)

            is_normal = self.normality_results[var]['is_normal']
            outlier_count = outlier_info['iqr_outliers']

            axes[row, col].set_title(f'{var}\nNormalny: {"Tak" if is_normal else "Nie"}, '
                                     f'Outliers: {outlier_count}')
            axes[row, col].set_xlabel(var)
            axes[row, col].set_ylabel('Gęstość')
            axes[row, col].grid(True, alpha=0.3)

            if i == 0:
                axes[row, col].legend()

        for j in range(len(self.quantitative_vars), 9):
            row = j // 3
            col = j % 3
            axes[row, col].remove()

        plt.tight_layout()
        plt.show()

        print("Wykresy rozkładów z outliers zostały wygenerowane")

    def create_scatter_plots(self):

        print("\n10. WYKRESY ROZRZUTU")
        print("-" * 30)

        correlation_matrix = self.df[self.all_numeric].corr()

        corr_pairs = []
        for i in range(len(self.all_numeric)):
            for j in range(i+1, len(self.all_numeric)):
                var1, var2 = self.all_numeric[i], self.all_numeric[j]
                corr_value = correlation_matrix.loc[var1, var2]
                corr_pairs.append((var1, var2, corr_value, abs(corr_value)))

        corr_pairs.sort(key=lambda x: x[3], reverse=True)
        self.top_correlations = corr_pairs[:3]

        print("NAJSILNIEJ SKORELOWANE PARY ZMIENNYCH:")
        for i, (var1, var2, corr, abs_corr) in enumerate(self.top_correlations, 1):
            print(f"   {i}. {var1} ↔ {var2}: r = {corr:.3f}")

        self._create_simple_scatter_plots()

        self._create_categorized_scatter_plots()

        print("✅ Wykresy rozrzutu zostały wygenerowane")

    def _create_simple_scatter_plots(self):

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Wykresy rozrzutu - najsilniej skorelowane pary', fontsize=16, fontweight='bold')

        for i, (var1, var2, corr, _) in enumerate(self.top_correlations):
            axes[i].scatter(self.df[var1], self.df[var2], alpha=0.6, color='steelblue')

            z = np.polyfit(self.df[var1].dropna(), self.df[var2].dropna(), 1)
            p = np.poly1d(z)
            axes[i].plot(self.df[var1], p(self.df[var1]), "r--", alpha=0.8, linewidth=2)

            axes[i].set_xlabel(var1)
            axes[i].set_ylabel(var2)
            axes[i].set_title(f'{var1} vs {var2}\nr = {corr:.3f}')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _create_categorized_scatter_plots(self):

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Wykresy rozrzutu skategoryzowane według płci i wyniku',
                     fontsize=16, fontweight='bold')

        for i, (var1, var2, corr, _) in enumerate(self.top_correlations):
            for gender, color, label in [(0, 'pink', 'Kobiety'), (1, 'lightblue', 'Mężczyźni')]:
                data_subset = self.df[self.df['Gender'] == gender]
                axes[0, i].scatter(data_subset[var1], data_subset[var2],
                                   alpha=0.6, color=color, label=label, s=50)

            axes[0, i].set_xlabel(var1)
            axes[0, i].set_ylabel(var2)
            axes[0, i].set_title(f'{var1} vs {var2} (według płci)\nr = {corr:.3f}')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)

        for i, (var1, var2, corr, _) in enumerate(self.top_correlations):
            for result, color, label in [('Negative', 'lightgreen', 'Negative'),
                                         ('Positive', 'lightcoral', 'Positive')]:
                data_subset = self.df[self.df['Result'] == result]
                axes[1, i].scatter(data_subset[var1], data_subset[var2],
                                   alpha=0.6, color=color, label=label, s=50)

            axes[1, i].set_xlabel(var1)
            axes[1, i].set_ylabel(var2)
            axes[1, i].set_title(f'{var1} vs {var2} (według wyniku)\nr = {corr:.3f}')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_analysis_summary(self):

        return {
            'variable_importance': self.importance_df,
            'normality_results': self.normality_results,
            'outliers_info': self.outliers_info,
            'top_correlations': self.top_correlations
        }