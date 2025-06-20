"""
MODUŁ TESTOWANIA HIPOTEZ
Testowanie trzech hipotez badawczych z różnymi zmiennymi zależnymi
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score, classification_report
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

class HypothesisTesting:
    """Klasa do testowania trzech hipotez badawczych"""

    def __init__(self, df_analysis):
        self.df = df_analysis

        # Definicje hipotez
        self.hypotheses = {
            'h1': {
                'title': 'Wiek pacjenta jest głównym predyktorem poziomu troponiny',
                'dependent': 'Troponin',
                'independent': ['Age', 'Gender', 'Heart rate'],
                'type': 'regression',
                'description': 'Analiza regresyjna - czy wiek, płeć i tętno przewidują poziom troponiny'
            },
            'h2': {
                'title': 'Płeć determinuje poziom ciśnienia skurczowego',
                'dependent': 'Systolic blood pressure',
                'independent': ['Gender', 'Age', 'Blood sugar'],
                'type': 'regression',
                'description': 'Analiza regresyjna - czy płeć, wiek i cukier przewidują ciśnienie'
            },
            'h3': {
                'title': 'Zawał serca można przewidzieć na podstawie biomarkerów',
                'dependent': 'Result_Binary',
                'independent': ['Troponin', 'CK-MB', 'Heart rate'],
                'type': 'classification',
                'description': 'Analiza klasyfikacyjna - czy biomarkery przewidują zawał'
            }
        }

        self.results = {}

    def test_all_hypotheses(self):
        """Testuje wszystkie trzy hipotezy"""

        print("\n" + "=" * 80)
        print("TESTOWANIE HIPOTEZ BADAWCZYCH")
        print("=" * 80)

        # Testuj każdą hipotezę
        for hyp_id in ['h1', 'h2', 'h3']:
            print(f"\n{'='*60}")
            print(f"HIPOTEZA {hyp_id.upper()}")
            print(f"{'='*60}")

            hyp = self.hypotheses[hyp_id]
            print(f"📌 {hyp['title']}")
            print(f"📊 {hyp['description']}")
            print(f"🎯 Zmienna zależna: {hyp['dependent']}")
            print(f"📈 Zmienne niezależne: {', '.join(hyp['independent'])}")

            if hyp['type'] == 'regression':
                result = self._test_regression_hypothesis(hyp_id, hyp)
            else:
                result = self._test_classification_hypothesis(hyp_id, hyp)

            self.results[hyp_id] = result

            # Wyświetl wyniki
            self._display_hypothesis_results(hyp_id, result)

        # Podsumowanie wszystkich hipotez
        self._create_final_summary()

        return self.results

    def _test_regression_hypothesis(self, hyp_id, hyp):
        """Testuje hipotezę regresyjną"""

        print(f"\n🔬 ANALIZA REGRESYJNA - {hyp_id.upper()}")
        print("-" * 40)

        # Przygotowanie danych
        X = self.df[hyp['independent']]
        y = self.df[hyp['dependent']]

        # Podstawowe statystyki
        print(f"📊 Statystyki zmiennej zależnej ({hyp['dependent']}):")
        print(f"   Średnia: {y.mean():.3f}")
        print(f"   Odchylenie std: {y.std():.3f}")
        print(f"   Zakres: [{y.min():.3f}, {y.max():.3f}]")

        # Korelacje
        print(f"\n🔗 Korelacje z zmienną zależną:")
        correlations = {}
        for var in hyp['independent']:
            corr, p_val = pearsonr(self.df[var], y)
            correlations[var] = {'correlation': corr, 'p_value': p_val}
            significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            print(f"   {var}: r = {corr:.3f} (p = {p_val:.4f}) {significance}")

        # Model regresji liniowej
        print(f"\n📈 MODEL REGRESJI LINIOWEJ:")

        # Podział danych
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Model
        reg_model = LinearRegression()
        reg_model.fit(X_train, y_train)

        # Predykcje
        y_pred_train = reg_model.predict(X_train)
        y_pred_test = reg_model.predict(X_test)

        # Metryki
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mse_test = np.mean((y_test - y_pred_test) ** 2)
        rmse_test = np.sqrt(mse_test)

        print(f"   R² (treningowy): {r2_train:.3f}")
        print(f"   R² (testowy): {r2_test:.3f}")
        print(f"   RMSE (testowy): {rmse_test:.3f}")

        # Współczynniki
        print(f"\n📊 Współczynniki regresji:")
        print(f"   Wyraz wolny: {reg_model.intercept_:.3f}")
        for i, var in enumerate(hyp['independent']):
            coef = reg_model.coef_[i]
            print(f"   {var}: {coef:.3f}")

        # Test istotności modelu (F-test)
        n = len(y_train)
        k = len(hyp['independent'])

        if r2_train < 1.0:  # Unikamy dzielenia przez zero
            f_statistic = (r2_train / k) / ((1 - r2_train) / (n - k - 1))
            f_p_value = 1 - stats.f.cdf(f_statistic, k, n - k - 1)

            print(f"\n🔬 Test istotności modelu (F-test):")
            print(f"   F-statystyka: {f_statistic:.3f}")
            print(f"   p-value: {f_p_value:.6f}")
            print(f"   Model {'ISTOTNY' if f_p_value < 0.05 else 'NIEISTOTNY'} statystycznie (α=0.05)")

        # Interpretacja wyników
        interpretation = self._interpret_regression_results(r2_test, correlations, hyp['dependent'])

        return {
            'type': 'regression',
            'correlations': correlations,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'rmse': rmse_test,
            'coefficients': dict(zip(['intercept'] + hyp['independent'],
                                     [reg_model.intercept_] + list(reg_model.coef_))),
            'f_statistic': f_statistic if 'f_statistic' in locals() else None,
            'f_p_value': f_p_value if 'f_p_value' in locals() else None,
            'interpretation': interpretation,
            'model': reg_model
        }

    def _test_classification_hypothesis(self, hyp_id, hyp):
        """Testuje hipotezę klasyfikacyjną"""

        print(f"\n🔬 ANALIZA KLASYFIKACYJNA - {hyp_id.upper()}")
        print("-" * 40)

        # Przygotowanie danych
        X = self.df[hyp['independent']]
        y = self.df[hyp['dependent']]

        # Podstawowe statystyki
        class_counts = y.value_counts()
        print(f"📊 Rozkład klasy docelowej:")
        print(f"   Klasa 0 (Negative): {class_counts[0]} ({class_counts[0]/len(y)*100:.1f}%)")
        print(f"   Klasa 1 (Positive): {class_counts[1]} ({class_counts[1]/len(y)*100:.1f}%)")

        # Korelacje z klasą docelową
        print(f"\n🔗 Korelacje z klasą docelową:")
        correlations = {}
        for var in hyp['independent']:
            corr, p_val = pearsonr(self.df[var], y)
            correlations[var] = {'correlation': corr, 'p_value': p_val}
            significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            print(f"   {var}: r = {corr:.3f} (p = {p_val:.4f}) {significance}")

        # Podział danych
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                            random_state=42, stratify=y)

        # Standaryzacja
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model regresji logistycznej
        print(f"\n📈 MODEL REGRESJI LOGISTYCZNEJ:")

        log_model = LogisticRegression(random_state=42)
        log_model.fit(X_train_scaled, y_train)

        # Predykcje
        y_pred = log_model.predict(X_test_scaled)
        y_proba = log_model.predict_proba(X_test_scaled)[:, 1]

        # Metryki
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(f"   Dokładność: {accuracy:.3f}")
        print(f"   AUC: {auc:.3f}")

        # Współczynniki
        print(f"\n📊 Współczynniki regresji logistycznej:")
        print(f"   Wyraz wolny: {log_model.intercept_[0]:.3f}")
        for i, var in enumerate(hyp['independent']):
            coef = log_model.coef_[0][i]
            odds_ratio = np.exp(coef)
            print(f"   {var}: {coef:.3f} (OR = {odds_ratio:.3f})")

        # Raport klasyfikacji
        print(f"\n📋 RAPORT KLASYFIKACJI:")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

        # Interpretacja wyników
        interpretation = self._interpret_classification_results(accuracy, auc, correlations)

        return {
            'type': 'classification',
            'correlations': correlations,
            'accuracy': accuracy,
            'auc': auc,
            'coefficients': dict(zip(['intercept'] + hyp['independent'],
                                     [log_model.intercept_[0]] + list(log_model.coef_[0]))),
            'odds_ratios': dict(zip(hyp['independent'], np.exp(log_model.coef_[0]))),
            'classification_report': classification_report(y_test, y_pred, target_names=['Negative', 'Positive'], output_dict=True),
            'interpretation': interpretation,
            'model': log_model,
            'scaler': scaler
        }

    def _interpret_regression_results(self, r2, correlations, dependent_var):
        """Interpretuje wyniki analizy regresyjnej"""

        # Siła modelu
        if r2 < 0.1:
            model_strength = "bardzo słaby"
        elif r2 < 0.3:
            model_strength = "słaby"
        elif r2 < 0.5:
            model_strength = "umiarkowany"
        elif r2 < 0.7:
            model_strength = "silny"
        else:
            model_strength = "bardzo silny"

        # Najsilniejszy predyktor
        strongest_predictor = max(correlations.keys(),
                                  key=lambda x: abs(correlations[x]['correlation']))
        strongest_corr = correlations[strongest_predictor]['correlation']

        return {
            'model_strength': model_strength,
            'r2_interpretation': f"Model wyjaśnia {r2*100:.1f}% wariancji {dependent_var}",
            'strongest_predictor': strongest_predictor,
            'strongest_correlation': strongest_corr,
            'conclusion': f"Model ma {model_strength} wpływ predykcyjny. "
                          f"Najsilniejszym predyktorem jest {strongest_predictor} (r={strongest_corr:.3f})"
        }

    def _interpret_classification_results(self, accuracy, auc, correlations):
        """Interpretuje wyniki analizy klasyfikacyjnej"""

        # Jakość klasyfikacji na podstawie AUC
        if auc < 0.6:
            model_quality = "bardzo słaba"
        elif auc < 0.7:
            model_quality = "słaba"
        elif auc < 0.8:
            model_quality = "dobra"
        elif auc < 0.9:
            model_quality = "bardzo dobra"
        else:
            model_quality = "doskonała"

        # Najsilniejszy predyktor
        strongest_predictor = max(correlations.keys(),
                                  key=lambda x: abs(correlations[x]['correlation']))
        strongest_corr = correlations[strongest_predictor]['correlation']

        return {
            'model_quality': model_quality,
            'auc_interpretation': f"AUC = {auc:.3f} wskazuje na {model_quality} zdolność predykcyjną",
            'accuracy_interpretation': f"Dokładność {accuracy*100:.1f}% oznacza poprawną klasyfikację",
            'strongest_predictor': strongest_predictor,
            'strongest_correlation': strongest_corr,
            'conclusion': f"Model ma {model_quality} zdolność predykcyjną. "
                          f"Najsilniejszym predyktorem jest {strongest_predictor} (r={strongest_corr:.3f})"
        }

    def _display_hypothesis_results(self, hyp_id, result):
        """Wyświetla wyniki testowania hipotezy"""

        print(f"\n💡 INTERPRETACJA WYNIKÓW - {hyp_id.upper()}:")
        print("-" * 50)

        interp = result['interpretation']
        print(f"📊 {interp['conclusion']}")

        if result['type'] == 'regression':
            print(f"📈 {interp['r2_interpretation']}")
            print(f"🎯 Model ma {interp['model_strength']} wpływ predykcyjny")

            # Wnioski o hipotezie
            r2 = result['r2_test']
            if r2 > 0.1 and any(abs(corr['correlation']) > 0.3 for corr in result['correlations'].values()):
                conclusion = "HIPOTEZA POTWIERDZONA CZĘŚCIOWO"
                explanation = f"Model wykazuje {interp['model_strength']} związek predykcyjny"
            elif r2 > 0.05:
                conclusion = "HIPOTEZA POTWIERDZONA SŁABO"
                explanation = "Istnieje słaby związek, ale wymaga dalszych badań"
            else:
                conclusion = "HIPOTEZA ODRZUCONA"
                explanation = "Brak znaczącego związku predykcyjnego"

        else:  # classification
            print(f"🎯 {interp['auc_interpretation']}")
            print(f"✅ {interp['accuracy_interpretation']}")

            # Wnioski o hipotezie
            auc = result['auc']
            accuracy = result['accuracy']
            if auc > 0.7 and accuracy > 0.7:
                conclusion = "HIPOTEZA POTWIERDZONA"
                explanation = f"Model ma {interp['model_quality']} zdolność predykcyjną"
            elif auc > 0.6:
                conclusion = "HIPOTEZA POTWIERDZONA CZĘŚCIOWO"
                explanation = "Model wykazuje umiarkowaną zdolność predykcyjną"
            else:
                conclusion = "HIPOTEZA ODRZUCONA"
                explanation = "Brak wystarczającej zdolności predykcyjnej"

        print(f"\n🏆 WNIOSEK: {conclusion}")
        print(f"📝 UZASADNIENIE: {explanation}")

        # Dodaj do wyników
        result['hypothesis_conclusion'] = conclusion
        result['hypothesis_explanation'] = explanation

    def _create_final_summary(self):
        """Tworzy podsumowanie wszystkich hipotez"""

        print(f"\n\n" + "="*80)
        print("PODSUMOWANIE TESTOWANIA HIPOTEZ")
        print("="*80)

        print(f"\n📋 STATUS HIPOTEZ:")
        for hyp_id in ['h1', 'h2', 'h3']:
            hyp = self.hypotheses[hyp_id]
            result = self.results[hyp_id]

            status_icon = "✅" if "POTWIERDZONA" in result['hypothesis_conclusion'] else "❌"
            print(f"{status_icon} {hyp_id.upper()}: {result['hypothesis_conclusion']}")
            print(f"   📌 {hyp['title']}")
            print(f"   📊 {result['hypothesis_explanation']}")

            if result['type'] == 'regression':
                print(f"   📈 R² = {result['r2_test']:.3f}")
            else:
                print(f"   📈 AUC = {result['auc']:.3f}, Accuracy = {result['accuracy']:.3f}")
            print()

        # Ranking predyktorów
        print(f"🏆 RANKING NAJSILNIEJSZYCH PREDYKTORÓW:")
        all_predictors = []

        for hyp_id, result in self.results.items():
            interp = result['interpretation']
            predictor = interp['strongest_predictor']
            strength = abs(interp['strongest_correlation'])
            all_predictors.append((predictor, strength, hyp_id))

        all_predictors.sort(key=lambda x: x[1], reverse=True)

        for i, (predictor, strength, hyp_id) in enumerate(all_predictors, 1):
            print(f"   {i}. {predictor}: |r| = {strength:.3f} (z hipotezy {hyp_id.upper()})")

        # Ogólne wnioski
        confirmed_hypotheses = sum(1 for result in self.results.values()
                                   if "POTWIERDZONA" in result['hypothesis_conclusion'])

        print(f"\n📊 STATYSTYKI OGÓLNE:")
        print(f"   Potwierdzone hipotezy: {confirmed_hypotheses}/3")
        print(f"   Procent sukcesu: {confirmed_hypotheses/3*100:.1f}%")

        if confirmed_hypotheses >= 2:
            overall_conclusion = "Badanie wykazało silne związki w analizowanych danych medycznych"
        elif confirmed_hypotheses == 1:
            overall_conclusion = "Badanie wykazało częściowe związki - wymaga dalszej analizy"
        else:
            overall_conclusion = "Badanie nie wykazało znaczących związków - konieczna reanaliza"

        print(f"\n🎯 WNIOSEK OGÓLNY:")
        print(f"   {overall_conclusion}")

    def get_detailed_results(self):
        """Zwraca szczegółowe wyniki wszystkich testów"""
        return {
            'hypotheses_definitions': self.hypotheses,
            'test_results': self.results,
            'summary': {
                'confirmed_count': sum(1 for result in self.results.values()
                                       if "POTWIERDZONA" in result['hypothesis_conclusion']),
                'total_count': len(self.results)
            }
        }