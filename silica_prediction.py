import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import clone


class SilicaPrediction:
    def __init__(self, data_path):
        self.data_path = data_path

    def avaliar_modelos(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        n_splits: int = 5
    ):
        """
        Avalia o desempenho de diferentes modelos de aprendizado de máquina.

        Parâmetros:
        X_train (pd.DataFrame): Conjunto de dados de treino (features).
        y_train (pd.Series): Conjunto de dados de treino (target).
        X_test (pd.DataFrame): Conjunto de dados de teste (features).
        y_test (pd.Series): Conjunto de dados de teste (target).
        n_splits (int): Número de divisões para validação cruzada temporal (TimeSeriesSplit).

        Retorna:
        pd.DataFrame: DataFrame contendo os resultados da avaliação dos modelos.
        """

        # Definir os modelos a serem avaliados
        modelos = {
            "Regressão Linear": LinearRegression()
        }

        resultados = []

        # Avaliação treino vs teste
        for nome, model in modelos.items():
            model = clone(model)
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)

            resultados.append({
                "Modelo": nome,
                "RMSE Treino": rmse_train,
                "RMSE Teste": rmse_test,
                "R2 Treino": r2_train,
                "R2 Teste": r2_test
            })

        df_resultados = pd.DataFrame(resultados)

        # Gráficos comparativos
        _, axes = plt.subplots(1, 2, figsize=(14, 5))

        df_resultados.plot(
            x="Modelo",
            y=["RMSE Treino", "RMSE Teste"],
            kind="bar",
            ax=axes[0]
        )
        axes[0].set_title("Comparação RMSE (Treino vs Teste)")
        axes[0].set_ylabel("RMSE")
        axes[0].grid(True)

        df_resultados.plot(
            x="Modelo",
            y=["R2 Treino", "R2 Teste"],
            kind="bar",
            ax=axes[1]
        )
        axes[1].set_title("Comparação R² (Treino vs Teste)")
        axes[1].set_ylabel("R²")
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

        # Diagnóstico Over/Underfitting
        print("\nDiagnóstico de Overfitting / Underfitting")
        for _, row in df_resultados.iterrows():
            diff = row["RMSE Teste"] - row["RMSE Treino"]

            if diff > 0.2 * row["RMSE Treino"]:
                status = "Overfitting"
            elif row["R2 Teste"] < 0.3:
                status = "Underfitting"
            else:
                status = "Bom ajuste"

            print(f"- {row['Modelo']}: {status}")

        # Recomendação de modelos não-lineares
        print("\nRecomendação de Modelos Não-Lineares")

        for _, row in df_resultados.iterrows():
            if row["R2 Teste"] < 0.7:
                print(
                    f"- {row['Modelo']}: "
                    "R² < 0.7 → vale testar RandomForest ou XGBoost"
                )
            else:
                print(
                    f"- {row['Modelo']}: "
                    "Modelo linear já captura bem o padrão"
                )

        return df_resultados

    def plot_correlation_heatmap(self, df, figsize=(18, 18), annot=True, save_path="correlation_heatmap.png", dpi=300):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.drop(columns=['date'])

        df = df.dropna()
        df = df.drop_duplicates()

        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.').astype('float32')

        for col in df.columns:
            upper_limit = df[col].quantile(0.95)
            lower_limit = df[col].quantile(0.05)
            df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])
            df[col] = np.where(df[col] < lower_limit, lower_limit, df[col])

        corr_matrix = df.corr()

        plt.figure(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            linewidths=0.003,
            linecolor='white',
            annot=annot,
            cmap='plasma'
        )
        plt.title("Matriz de Correlação das Variáveis", fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=45, ha='right', fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.show()

    def load_data(self, clean_data=True):
        df = pd.read_csv(self.data_path)

        df = df.dropna()
        df = df.drop_duplicates()

        for col in df.columns:
            if col == 'date':
                continue
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.').astype('float32')

        for col in df.columns:
            if col == 'date':
                continue
            upper_limit = df[col].quantile(0.95)
            lower_limit = df[col].quantile(0.05)
            df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])
            df[col] = np.where(df[col] < lower_limit, lower_limit, df[col])

        if clean_data:
            df = df.drop(columns=['% Iron Concentrate', '% Iron Feed'])
            airflow_stage_1 = [
                'Flotation Column 01 Air Flow',
                'Flotation Column 02 Air Flow',
                'Flotation Column 03 Air Flow',
                'Flotation Column 04 Air Flow'
            ]

            airflow_stage_2 = [
                'Flotation Column 06 Air Flow',
                'Flotation Column 07 Air Flow'
            ]

            df['airflow_stage_1'] = df[airflow_stage_1].mean(axis=1)
            df['airflow_stage_2'] = df[airflow_stage_2].mean(axis=1)
            df = df.drop(columns=airflow_stage_1 + airflow_stage_2)

            level_stage_1 = [
                'Flotation Column 01 Level',
                'Flotation Column 02 Level',
                'Flotation Column 03 Level'
            ]

            level_stage_2 = [
                'Flotation Column 05 Level',
                'Flotation Column 06 Level',
                'Flotation Column 07 Level'
            ]

            df['level_stage_1'] = df[level_stage_1].mean(axis=1)
            df['level_stage_2'] = df[level_stage_2].mean(axis=1)
            df = df.drop(columns=level_stage_1 + level_stage_2)

        return df

    def preprocess_data(self, df, split_perc=0.8):
        y = df['% Silica Concentrate']
        X = df.drop(['% Silica Concentrate'], axis=1)

        df = df.sort_values("date")

        start_date = df["date"].iloc[0]
        split_point = int(len(df) * split_perc)
        split_date = df["date"].iloc[split_point]
        end_date = df["date"].iloc[-1]

        print(f"Start Date: {start_date}, Split Date: {split_date}, End Date: {end_date}")

        train = df.iloc[:split_point]
        test = df.iloc[split_point:]

        X_train = train.drop(columns=["% Silica Concentrate", "date"])
        y_train = train["% Silica Concentrate"]

        X_test = test.drop(columns=["% Silica Concentrate", "date"])
        y_test = test["% Silica Concentrate"]

        train = train.drop(columns=["date"])
        test = test.drop(columns=["date"])

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)

        y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

        return X_train, y_train, X_test, y_test

    def train_model(self, X_train, y_train, X_test, y_test):
        model = LinearRegression()
        model.fit(X_train, y_train)

        r_squared = model.score(X_test, y_test)
        print(f"R-squared score: {r_squared}")

        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)

        return model, rmse, y_pred

    def pipeline_lr(self, fi=1, cd=True):
        print("Carregando dados...")
        df = self.load_data(clean_data=cd)

        print("Pré-processando dados...")
        X_train, y_train, X_test, y_test = self.preprocess_data(df)

        print("Treinando modelo...")
        model, rmse, y_pred = self.train_model(X_train, y_train, X_test, y_test)

        if fi > 1:
            model, rmse, y_pred = self.train_model(X_train, y_train, X_test, y_test)

            feature_importances = pd.DataFrame({
                'Feature': df.drop(columns=["% Silica Concentrate", "date"]).columns,
                'Importance': model.coef_ if hasattr(model, 'coef_') else model.feature_importances_
            })

            feature_importances['Importance'] = np.abs(feature_importances['Importance'])
            feature_importances['Importance'] = (
                feature_importances['Importance'] /
                feature_importances['Importance'].sum()
            ) * 100
            feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='plasma')
            plt.title('Importância das Features (Modelo Completo)')
            plt.xlabel('Importância (%)')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.show()

            top_features = feature_importances.head(fi)['Feature']
            X_train_top = pd.DataFrame(X_train, columns=df.drop(columns=["% Silica Concentrate", "date"]).columns)[top_features]
            X_test_top = pd.DataFrame(X_test, columns=df.drop(columns=["% Silica Concentrate", "date"]).columns)[top_features]

            print(f"Recalculando o modelo com as {fi} features mais importantes...")
            model_top, rmse, y_pred = self.train_model(X_train_top, y_train, X_test_top, y_test)

            model = model_top

            baseline_pred = np.full_like(y_test, y_train.mean())
            baseline_rmse = np.sqrt(np.mean((y_test - baseline_pred) ** 2))

            gain = 1 - (rmse / baseline_rmse)

            std_target = y_test.std()
            mean_target = y_test.mean()
            rmse_ratio = rmse / std_target

            print(f"Modelo Avaliado com {fi} features:")
            print(f"- RMSE (Erro Quadrático Médio): {rmse:.4f}")
            print(f"- Desvio Padrão do Alvo: {std_target:.4f}")
            print(f"- Razão RMSE / Desvio Padrão do Alvo: {rmse_ratio:.2f}")

            if rmse_ratio < 0.5:
                print("  -> Excelente: O erro está bem abaixo da variabilidade natural dos dados.")
            elif rmse_ratio < 1:
                print("  -> Bom: O erro é comparável à variabilidade dos dados.")
            else:
                print("  -> Fraco: O erro é maior que a variabilidade do alvo. O modelo precisa de melhorias.")

            print(f"- Ganho em relação ao baseline: {gain:.2%}")
            if gain > 0.50:
                print("  -> Excelente: O modelo é significativamente melhor que o baseline.")
            elif gain > 0.20:
                print("  -> Bom: O modelo é razoavelmente melhor que o baseline.")
            elif gain > 0:
                print("  -> Marginal: O modelo é apenas ligeiramente melhor que o baseline.")
            else:
                print("  -> Alerta: O modelo é pior que o baseline.")

            X_train = X_train_top
            X_test = X_test_top

        else:
            baseline_pred = np.full_like(y_test, y_train.mean())
            baseline_rmse = np.sqrt(np.mean((y_test - baseline_pred) ** 2))

            gain = 1 - (rmse / baseline_rmse)

            std_target = y_test.std()
            mean_target = y_test.mean()
            rmse_ratio = rmse / std_target

            print(f"Modelo Avaliado com todas as features:")
            print(f"- RMSE (Erro Quadrático Médio): {rmse:.4f}")
            print(f"- Desvio Padrão do Alvo: {std_target:.4f}")
            print(f"- Razão RMSE / Desvio Padrão do Alvo: {rmse_ratio:.2f}")

            if rmse_ratio < 0.5:
                print("  -> Excelente: O erro está bem abaixo da variabilidade natural dos dados.")
            elif rmse_ratio < 1:
                print("  -> Bom: O erro é comparável à variabilidade dos dados.")
            else:
                print("  -> Fraco: O erro é maior que a variabilidade do alvo. O modelo precisa de melhorias.")

            print(f"- Ganho em relação ao baseline: {gain:.2%}")
            if gain > 0.50:
                print("  -> Excelente: O modelo é significativamente melhor que o baseline.")
            elif gain > 0.20:
                print("  -> Bom: O modelo é razoavelmente melhor que o baseline.")
            elif gain > 0:
                print("  -> Marginal: O modelo é apenas ligeiramente melhor que o baseline.")
            else:
                print("  -> Alerta: O modelo é pior que o baseline.")

        self.avaliar_modelos(X_train, y_train, X_test, y_test)

        return model, rmse, y_pred