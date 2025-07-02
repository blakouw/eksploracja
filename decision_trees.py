import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             r2_score, mean_squared_error, mean_absolute_error)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DecisionTreeAnalysis:

    def __init__(self, df_analysis):
        self.df = df_analysis

        self.hypotheses = {
            'h1': {
                'title': 'Wiek jako predyktor poziomu troponiny',
                'dependent': 'Troponin',
                'independent': ['Age', 'Gender', 'Heart rate'],
                'type': 'regression',
                'description': 'Drzewo regresyjne - przewidywanie poziomu troponiny'
            },
            'h2': {
                'title': 'Płeć jako determinanta ciśnienia skurczowego',
                'dependent': 'Systolic blood pressure',
                'independent': ['Gender', 'Age', 'Blood sugar'],
                'type': 'regression',
                'description': 'Drzewo regresyjne - przewidywanie ciśnienia skurczowego'
            },
            'h3': {
                'title': 'Biomarkery jako predyktory zawału serca',
                'dependent': 'Result_Binary',
                'independent': ['Troponin', 'CK-MB', 'Heart rate'],
                'type': 'classification',
                'description': 'Drzewo klasyfikacyjne - przewidywanie zawału serca'
            }
        }

        self.results = {}
        self.models = {}
        self.rules = {}

    def run_all_tree_analyses(self):
        """Uruchamia analizę drzew decyzyjnych dla wszystkich hipotez"""

        print("\n" + "=" * 80)
        print("CZĘŚĆ III: INDUKCJA DRZEW DECYZYJNYCH")
        print("=" * 80)

        for hyp_id in ['h1', 'h2', 'h3']:
            print(f"\n{'='*60}")
            print(f"ANALIZA DRZEWA - HIPOTEZA {hyp_id.upper()}")
            print(f"{'='*60}")

            hyp = self.hypotheses[hyp_id]
            print(f"{hyp['title']}")
            print(f"{hyp['description']}")
            print(f"Zmienna zależna: {hyp['dependent']}")
            print(f"Zmienne niezależne: {', '.join(hyp['independent'])}")

            if hyp['type'] == 'regression':
                result = self._build_regression_tree(hyp_id, hyp)
            else:
                result = self._build_classification_tree(hyp_id, hyp)

            self.results[hyp_id] = result

            self._visualize_tree(hyp_id, hyp, result['model'])

            self._plot_feature_importance(hyp_id, hyp, result['model'])

            self._formalize_rules(hyp_id, hyp, result['model'])

            self._evaluate_model(hyp_id, hyp, result)

        self._create_trees_summary()

    def _build_regression_tree(self, hyp_id, hyp):
        """Buduje drzewo regresyjne"""

        print(f"\n🌳 BUDOWANIE DRZEWA REGRESYJNEGO - {hyp_id.upper()}")
        print("-" * 50)

        X = self.df[hyp['independent']].copy()
        y = self.df[hyp['dependent']].copy()

        print(f"📊 Rozmiar danych: {X.shape[0]} próbek, {X.shape[1]} cech")
        print(f"📈 Zmienna docelowa: {hyp['dependent']}")
        print(f"   Średnia: {y.mean():.3f}")
        print(f"   Odchylenie std: {y.std():.3f}")
        print(f"   Zakres: [{y.min():.3f}, {y.max():.3f}]")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        tree_model = DecisionTreeRegressor(
            max_depth=5,
            min_samples_split=100,
            min_samples_leaf=50,
            random_state=42
        )

        tree_model.fit(X_train, y_train)


        y_pred_train = tree_model.predict(X_train)
        y_pred_test = tree_model.predict(X_test)

        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)

        print(f"\nMETRYKI DRZEWA REGRESYJNEGO:")
        print(f"   R² (treningowy): {r2_train:.4f}")
        print(f"   R² (testowy): {r2_test:.4f}")
        print(f"   RMSE: {rmse_test:.4f}")
        print(f"   MAE: {mae_test:.4f}")
        print(f"   Liczba liści: {tree_model.get_n_leaves()}")
        print(f"   Głębokość drzewa: {tree_model.get_depth()}")

        self.models[hyp_id] = tree_model

        return {
            'model': tree_model,
            'type': 'regression',
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mse': mse_test,
            'mae': mae_test,
            'rmse': rmse_test,
            'feature_names': list(X.columns)
        }

    def _build_classification_tree(self, hyp_id, hyp):

        print(f"\nBUDOWANIE DRZEWA KLASYFIKACYJNEGO - {hyp_id.upper()}")
        print("-" * 50)

        # Przygotowanie danych
        X = self.df[hyp['independent']].copy()
        y = self.df[hyp['dependent']].copy()

        print(f"Rozmiar danych: {X.shape[0]} próbek, {X.shape[1]} cech")
        print(f"Zmienna docelowa: {hyp['dependent']}")

        class_distribution = y.value_counts()
        print(f"   Rozkład klas:")
        for cls, count in class_distribution.items():
            print(f"     Klasa {cls}: {count} ({count/len(y)*100:.1f}%)")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        tree_model = DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=80,
            min_samples_leaf=30,
            random_state=42,
            class_weight='balanced'
        )

        tree_model.fit(X_train, y_train)

        y_pred_train = tree_model.predict(X_train)
        y_pred_test = tree_model.predict(X_test)
        y_proba_test = tree_model.predict_proba(X_test)

        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_test = accuracy_score(y_test, y_pred_test)

        print(f"\n📈 METRYKI DRZEWA KLASYFIKACYJNEGO:")
        print(f"   Dokładność (treningowa): {accuracy_train:.4f}")
        print(f"   Dokładność (testowa): {accuracy_test:.4f}")
        print(f"   Liczba liści: {tree_model.get_n_leaves()}")
        print(f"   Głębokość drzewa: {tree_model.get_depth()}")

        print(f"\nRAPORT KLASYFIKACJI:")
        class_names = ['Negative', 'Positive']
        print(classification_report(y_test, y_pred_test, target_names=class_names))

        # Zapisz model
        self.models[hyp_id] = tree_model

        return {
            'model': tree_model,
            'type': 'classification',
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'y_proba_test': y_proba_test,
            'accuracy_train': accuracy_train,
            'accuracy_test': accuracy_test,
            'classification_report': classification_report(y_test, y_pred_test,
                                                           target_names=class_names,
                                                           output_dict=True),
            'feature_names': list(X.columns)
        }

    def _visualize_tree(self, hyp_id, hyp, model):


        print(f"\nGenerowanie wizualizacji drzewa...")

        plt.figure(figsize=(20, 12))

        class_names = None
        if hyp['type'] == 'classification':
            class_names = ['Negative', 'Positive']

        # Rysowanie drzewa
        plot_tree(
            model,
            feature_names=hyp['independent'],
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=10,
            max_depth=3
        )

        plt.title(f'Drzewo {hyp["type"]} - {hyp["title"]}',
                  fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()

        print("Wizualizacja drzewa została wygenerowana")

    def _plot_feature_importance(self, hyp_id, hyp, model):

        print(f"\nAnaliza ważności predyktorów...")

        feature_importance = model.feature_importances_
        feature_names = hyp['independent']

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True)

        print(f"📈 Ranking ważności cech:")
        for _, row in importance_df.iterrows():
            print(f"   {row['Feature']}: {row['Importance']:.4f}")


        plt.figure(figsize=(10, 6))

        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
        bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors)

        plt.title(f'Ważność predyktorów - {hyp["title"]}', fontweight='bold')
        plt.xlabel('Ważność')
        plt.ylabel('Predyktory')

        # Dodaj wartości na słupkach
        for bar, importance in zip(bars, importance_df['Importance']):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{importance:.3f}', ha='left', va='center')

        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()

        print("✅ Wykres ważności cech został wygenerowany")

        return importance_df

    def _formalize_rules(self, hyp_id, hyp, model):

        print(f"\n📋 FORMALIZACJA REGUŁ DRZEWA - {hyp_id.upper()}")
        print("-" * 50)

        tree_text = export_text(
            model,
            feature_names=hyp['independent'],
            max_depth=4,
            spacing=3,
            decimals=2,
            show_weights=True
        )

        tree = model.tree_
        feature_names = hyp['independent']

        rules = []

        def extract_rules_recursive(node_id, conditions, depth=0):

            if depth > 4:
                return

            if tree.children_left[node_id] == tree.children_right[node_id]:
                n_samples = tree.n_node_samples[node_id]
                value = tree.value[node_id]

                if hyp['type'] == 'regression':
                    predicted_value = value[0][0]
                    impurity = tree.impurity[node_id]  # MSE dla regresji

                    rule = {
                        'conditions': conditions.copy(),
                        'node_id': node_id,
                        'n_samples': n_samples,
                        'predicted_value': predicted_value,
                        'impurity': impurity,
                        'support': n_samples / tree.n_node_samples[0],
                        'type': 'regression'
                    }
                else:
                    # Klasyfikacja
                    class_counts = value[0]
                    predicted_class = np.argmax(class_counts)
                    total_samples = np.sum(class_counts)
                    confidence = class_counts[predicted_class] / total_samples if total_samples > 0 else 0

                    rule = {
                        'conditions': conditions.copy(),
                        'node_id': node_id,
                        'n_samples': n_samples,
                        'predicted_class': predicted_class,
                        'class_distribution': class_counts,
                        'confidence': confidence,
                        'support': n_samples / tree.n_node_samples[0],
                        'type': 'classification'
                    }

                rules.append(rule)
                return

            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            feature_name = feature_names[feature_idx]

            left_conditions = conditions.copy()
            left_conditions.append(f"{feature_name} <= {threshold:.2f}")
            extract_rules_recursive(tree.children_left[node_id], left_conditions, depth + 1)

            right_conditions = conditions.copy()
            right_conditions.append(f"{feature_name} > {threshold:.2f}")
            extract_rules_recursive(tree.children_right[node_id], right_conditions, depth + 1)


        extract_rules_recursive(0, [])

        rules.sort(key=lambda x: x['support'], reverse=True)

        top_rules = rules[:5]

        print(f"🎯 NAJWAŻNIEJSZE REGUŁY (Top {len(top_rules)}):")

        for i, rule in enumerate(top_rules, 1):
            print(f"\n📌 REGUŁA {i}:")
            print(f"   Warunki: {' AND '.join(rule['conditions'])}")

            if rule['type'] == 'regression':
                print(f"   Przewidywana wartość: {rule['predicted_value']:.3f}")
                print(f"   Wariancja w liściu: {rule['impurity']:.3f}")
                print(f"   Wsparcie: {rule['support']:.3f} ({rule['n_samples']} próbek)")

                if rule['impurity'] < np.var(self.df[hyp['dependent']]) * 0.5:
                    quality = "DOBRA (niska wariancja)"
                elif rule['impurity'] < np.var(self.df[hyp['dependent']]):
                    quality = "UMIARKOWANA"
                else:
                    quality = "SŁABA (wysoka wariancja)"

                print(f"   Jakość reguły: {quality}")

            else:
                predicted_class_name = 'Positive' if rule['predicted_class'] == 1 else 'Negative'
                print(f"   Przewidywana klasa: {predicted_class_name}")
                print(f"   Pewność: {rule['confidence']:.3f}")
                print(f"   Wsparcie: {rule['support']:.3f} ({rule['n_samples']} próbek)")
                print(f"   Rozkład klas: {rule['class_distribution']}")

                if rule['confidence'] >= 0.8:
                    quality = "BARDZO DOBRA (wysoka pewność)"
                elif rule['confidence'] >= 0.6:
                    quality = "DOBRA"
                else:
                    quality = "SŁABA (niska pewność)"

                print(f"   Jakość reguły: {quality}")

        self.rules[hyp_id] = top_rules

        print("Reguły zostały sformalizowane")

        return top_rules

    def _evaluate_model(self, hyp_id, hyp, result):

        print(f"\n🔍 OCENA MODELU DRZEWA - {hyp_id.upper()}")
        print("-" * 40)

        if result['type'] == 'regression':
            self._evaluate_regression_tree(hyp_id, hyp, result)
        else:
            self._evaluate_classification_tree(hyp_id, hyp, result)

    def _evaluate_regression_tree(self, hyp_id, hyp, result):

        print("OCENA DRZEWA REGRESYJNEGO:")

        r2 = result['r2_test']
        print(f"   R² = {r2:.4f}")

        if r2 >= 0.7:
            interpretation = "Bardzo dobry model (wyjaśnia >70% wariancji)"
        elif r2 >= 0.5:
            interpretation = "Dobry model (wyjaśnia 50-70% wariancji)"
        elif r2 >= 0.3:
            interpretation = "Umiarkowany model (wyjaśnia 30-50% wariancji)"
        elif r2 >= 0.1:
            interpretation = "Słaby model (wyjaśnia 10-30% wariancji)"
        else:
            interpretation = "Bardzo słaby model (wyjaśnia <10% wariancji)"

        print(f"   Interpretacja: {interpretation}")

        print(f"   RMSE = {result['rmse']:.4f}")
        print(f"   MAE = {result['mae']:.4f}")

        residuals = result['y_test'] - result['y_pred_test']
        print(f"   Średnie reszty: {np.mean(residuals):.4f}")
        print(f"   Std reszt: {np.std(residuals):.4f}")

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(result['y_pred_test'], residuals, alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Przewidywane wartości')
        plt.ylabel('Reszty')
        plt.title('Wykres reszt')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.scatter(result['y_test'], result['y_pred_test'], alpha=0.6)
        min_val = min(result['y_test'].min(), result['y_pred_test'].min())
        max_val = max(result['y_test'].max(), result['y_pred_test'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'red', linestyle='--')
        plt.xlabel('Rzeczywiste wartości')
        plt.ylabel('Przewidywane wartości')
        plt.title('Rzeczywiste vs Przewidywane')
        plt.grid(True, alpha=0.3)

        plt.suptitle(f'Ocena drzewa regresyjnego - {hyp["title"]}')
        plt.tight_layout()
        plt.show()

    def _evaluate_classification_tree(self, hyp_id, hyp, result):

        print("OCENA DRZEWA KLASYFIKACYJNEGO:")

        accuracy = result['accuracy_test']
        print(f"   Dokładność = {accuracy:.4f}")

        if accuracy >= 0.9:
            interpretation = "Doskonały model (>90% dokładności)"
        elif accuracy >= 0.8:
            interpretation = "Bardzo dobry model (80-90% dokładności)"
        elif accuracy >= 0.7:
            interpretation = "Dobry model (70-80% dokładności)"
        elif accuracy >= 0.6:
            interpretation = "Umiarkowany model (60-70% dokładności)"
        else:
            interpretation = "Słaby model (<60% dokładności)"

        print(f"   Interpretacja: {interpretation}")

        cm = confusion_matrix(result['y_test'], result['y_pred_test'])

        print(f"\nMACIERZ POMYŁEK:")
        print(f"   True Negative:  {cm[0,0]}")
        print(f"   False Positive: {cm[0,1]}")
        print(f"   False Negative: {cm[1,0]}")
        print(f"   True Positive:  {cm[1,1]}")

        total_error = (cm[0,1] + cm[1,0]) / np.sum(cm)
        print(f"   Całkowity błąd drzewa: {total_error:.4f} ({total_error*100:.1f}%)")

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title(f'Macierz pomyłek - {hyp["title"]}')
        plt.xlabel('Przewidywane')
        plt.ylabel('Rzeczywiste')
        plt.tight_layout()
        plt.show()

    def _create_trees_summary(self):
        """Tworzy podsumowanie wszystkich drzew"""

        print(f"\n\n" + "="*80)
        print("PODSUMOWANIE ANALIZY DRZEW DECYZYJNYCH")
        print("="*80)

        summary_data = []

        for hyp_id in ['h1', 'h2', 'h3']:
            hyp = self.hypotheses[hyp_id]
            result = self.results[hyp_id]
            model = result['model']

            if result['type'] == 'regression':
                main_metric = f"R² = {result['r2_test']:.3f}"
                quality = self._interpret_r2(result['r2_test'])
            else:
                main_metric = f"Acc = {result['accuracy_test']:.3f}"
                quality = self._interpret_accuracy(result['accuracy_test'])

            summary_data.append({
                'Hipoteza': hyp_id.upper(),
                'Typ': result['type'],
                'Zmienna zależna': hyp['dependent'],
                'Liczba cech': len(hyp['independent']),
                'Głębokość': model.get_depth(),
                'Liczba liści': model.get_n_leaves(),
                'Główna metryka': main_metric,
                'Jakość': quality
            })

        summary_df = pd.DataFrame(summary_data)
        print("\n📊 TABELA PORÓWNAWCZA DRZEW:")
        print(summary_df.to_string(index=False))

        print(f"\nRANKING JAKOŚCI MODELI:")

        scores = []
        for hyp_id in ['h1', 'h2', 'h3']:
            result = self.results[hyp_id]
            if result['type'] == 'regression':
                score = max(0, result['r2_test'])
            else:
                score = result['accuracy_test']
            scores.append((hyp_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        for i, (hyp_id, score) in enumerate(scores, 1):
            hyp = self.hypotheses[hyp_id]
            print(f"   {i}. {hyp_id.upper()}: {hyp['title']}")
            print(f"      Score: {score:.3f}")

        print(f"\nWNIOSKI OGÓLNE:")

        best_hyp_id, best_score = scores[0]
        best_hyp = self.hypotheses[best_hyp_id]

        print(f"   • Najlepszy model: {best_hyp_id.upper()} ({best_hyp['title']})")
        print(f"   • Najwyższy score: {best_score:.3f}")

        # Analiza ważności cech
        print(f"   • Najważniejsze predyktory:")

        all_importances = {}
        for hyp_id in ['h1', 'h2', 'h3']:
            model = self.results[hyp_id]['model']
            feature_names = self.results[hyp_id]['feature_names']

            for feature, importance in zip(feature_names, model.feature_importances_):
                if feature not in all_importances:
                    all_importances[feature] = []
                all_importances[feature].append(importance)

        avg_importances = {
            feature: np.mean(importances)
            for feature, importances in all_importances.items()
        }

        sorted_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)

        for i, (feature, avg_importance) in enumerate(sorted_features[:5], 1):
            print(f"     {i}. {feature}: {avg_importance:.3f}")

        print(f"\nAnaliza drzew decyzyjnych została zakończona")

    def _interpret_r2(self, r2):
        if r2 >= 0.7:
            return "Bardzo dobra"
        elif r2 >= 0.5:
            return "Dobra"
        elif r2 >= 0.3:
            return "Umiarkowana"
        elif r2 >= 0.1:
            return "Słaba"
        else:
            return "Bardzo słaba"

    def _interpret_accuracy(self, accuracy):
        if accuracy >= 0.9:
            return "Doskonała"
        elif accuracy >= 0.8:
            return "Bardzo dobra"
        elif accuracy >= 0.7:
            return "Dobra"
        elif accuracy >= 0.6:
            return "Umiarkowana"
        else:
            return "Słaba"

    def generate_cost_sequence_plots(self):

        print(f"\nGenerowanie wykresów sekwencji kosztów...")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Wykresy sekwencji kosztów (Cost Sequence)', fontsize=16, fontweight='bold')

        for i, hyp_id in enumerate(['h1', 'h2', 'h3']):
            hyp = self.hypotheses[hyp_id]
            result = self.results[hyp_id]

            depths = range(1, 11)
            train_scores = []
            test_scores = []

            X_train = result['X_train']
            X_test = result['X_test']
            y_train = result['y_train']
            y_test = result['y_test']

            for depth in depths:
                if hyp['type'] == 'regression':
                    temp_model = DecisionTreeRegressor(max_depth=depth, random_state=42)
                else:
                    temp_model = DecisionTreeClassifier(max_depth=depth, random_state=42)

                temp_model.fit(X_train, y_train)

                if hyp['type'] == 'regression':
                    train_score = temp_model.score(X_train, y_train)
                    test_score = temp_model.score(X_test, y_test)
                else:
                    train_score = temp_model.score(X_train, y_train)
                    test_score = temp_model.score(X_test, y_test)

                train_scores.append(train_score)
                test_scores.append(test_score)

            # Plot
            axes[i].plot(depths, train_scores, 'b-o', label='Zbiór treningowy', markersize=4)
            axes[i].plot(depths, test_scores, 'r-o', label='Zbiór testowy', markersize=4)
            axes[i].set_title(f'{hyp_id.upper()}: {hyp["dependent"]}')
            axes[i].set_xlabel('Głębokość drzewa')
            axes[i].set_ylabel('Score')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, 1)

        plt.tight_layout()
        plt.show()

        print("Wykresy sekwencji kosztów zostały wygenerowane")

    def create_rules_summary_table(self):

        print(f"\nTABELA PODSUMOWUJĄCA REGUŁY:")
        print("-" * 70)

        rules_summary = []

        for hyp_id in ['h1', 'h2', 'h3']:
            hyp = self.hypotheses[hyp_id]
            rules = self.rules.get(hyp_id, [])

            for i, rule in enumerate(rules[:3], 1):  # Top 3 reguły
                rule_text = ' AND '.join(rule['conditions'][:2])  # Pierwsze 2 warunki
                if len(rule['conditions']) > 2:
                    rule_text += "..."

                if rule['type'] == 'regression':
                    outcome = f"{rule['predicted_value']:.2f}"
                    quality_metric = f"Var: {rule['impurity']:.3f}"
                else:
                    class_name = 'Positive' if rule['predicted_class'] == 1 else 'Negative'
                    outcome = class_name
                    quality_metric = f"Conf: {rule['confidence']:.3f}"

                rules_summary.append({
                    'Hipoteza': hyp_id.upper(),
                    'Reguła': f"R{i}",
                    'Warunki': rule_text,
                    'Wynik': outcome,
                    'Wsparcie': f"{rule['support']:.3f}",
                    'Jakość': quality_metric
                })

        rules_df = pd.DataFrame(rules_summary)
        print(rules_df.to_string(index=False))

        return rules_df

    def get_detailed_results(self):

        return {
            'hypotheses': self.hypotheses,
            'models': self.models,
            'results': self.results,
            'rules': self.rules,
            'summary': {
                'total_trees': len(self.models),
                'regression_trees': len([r for r in self.results.values() if r['type'] == 'regression']),
                'classification_trees': len([r for r in self.results.values() if r['type'] == 'classification'])
            }
        }

    def save_tree_visualizations(self, output_dir='tree_plots'):

        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Zapisywanie wizualizacji drzew do katalogu: {output_dir}")

        for hyp_id in ['h1', 'h2', 'h3']:
            hyp = self.hypotheses[hyp_id]
            model = self.models[hyp_id]

            plt.figure(figsize=(20, 12))

            class_names = None
            if hyp['type'] == 'classification':
                class_names = ['Negative', 'Positive']

            plot_tree(
                model,
                feature_names=hyp['independent'],
                class_names=class_names,
                filled=True,
                rounded=True,
                fontsize=10,
                max_depth=3
            )

            plt.title(f'Drzewo {hyp["type"]} - {hyp["title"]}',
                      fontsize=16, fontweight='bold', pad=20)

            plt.savefig(f'{output_dir}/tree_{hyp_id}.png', dpi=300, bbox_inches='tight')
            plt.close()

            print(f" Zapisano: tree_{hyp_id}.png")

        print(f"Zapisywanie wizualizacji zakończone")

    def export_rules_to_text(self, output_file='tree_rules.txt'):

        print(f"📄 Eksportowanie reguł do pliku: {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("REGUŁY DRZEW DECYZYJNYCH\n")
            f.write("=" * 50 + "\n\n")

            for hyp_id in ['h1', 'h2', 'h3']:
                hyp = self.hypotheses[hyp_id]
                rules = self.rules.get(hyp_id, [])

                f.write(f"HIPOTEZA {hyp_id.upper()}: {hyp['title']}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Typ: {hyp['type']}\n")
                f.write(f"Zmienna zależna: {hyp['dependent']}\n")
                f.write(f"Zmienne niezależne: {', '.join(hyp['independent'])}\n\n")

                for i, rule in enumerate(rules, 1):
                    f.write(f"REGUŁA {i}:\n")
                    f.write(f"Warunki: {' AND '.join(rule['conditions'])}\n")

                    if rule['type'] == 'regression':
                        f.write(f"Przewidywana wartość: {rule['predicted_value']:.3f}\n")
                        f.write(f"Wariancja w liściu: {rule['impurity']:.3f}\n")
                    else:
                        class_name = 'Positive' if rule['predicted_class'] == 1 else 'Negative'
                        f.write(f"Przewidywana klasa: {class_name}\n")
                        f.write(f"Pewność: {rule['confidence']:.3f}\n")

                    f.write(f"Wsparcie: {rule['support']:.3f} ({rule['n_samples']} próbek)\n")
                    f.write("\n")

                f.write("\n" + "="*50 + "\n\n")

        print(f"Reguły zostały wyeksportowane do pliku: {output_file}")

def run_decision_tree_analysis(df_analysis):

    print("Uruchamianie analizy drzew decyzyjnych...")

    tree_analysis = DecisionTreeAnalysis(df_analysis)

    tree_analysis.run_all_tree_analyses()

    tree_analysis.generate_cost_sequence_plots()

    tree_analysis.create_rules_summary_table()

    tree_analysis.export_rules_to_text()

    print("\nAnaliza drzew decyzyjnych została zakończona!")

    return tree_analysis