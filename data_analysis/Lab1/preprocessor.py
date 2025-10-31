import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MultiLabelBinarizer


class Preprocessor(TransformerMixin):
    numeric_features = ["Возраст", "Рост, см", "Вес, кг"]
    missing = "не указано"
    categorical_features = {
        "Семейное положение": [
            "не состою в браке",
            "состою в браке, живем порознь",
            "состою в браке",
            missing,
        ],
        "Дети": [
            "есть, живем порознь",
            "есть, уже взрослые",
            "есть, живем вместе",
            "нет, но хотелось бы",
            "нет",
            missing,
        ],
        "Курение": ["не курю", "редко", "курю", missing],
        "Алкоголь": [
            "изредка в компаниях",
            "не пью вообще",
            "люблю выпить",
            missing,
        ],
        "Доход": [
            "стабильный средний доход",
            "хорошо зарабатываю / обеспечен",
            "постоянный небольшой доход",
            "непостоянные заработки",
            missing,
        ],
        "Проживание": [
            "отдельная квартира",
            "нет постоянного жилья",
            "комната",
            "живу с приятелем / с подругой",
            "живу с родителями",
            "живу с партнером или с супругой",
            missing,
        ],
        "Наличие автомобиля": ["есть", "нет", missing],
        "Тело": [
            "плотное",
            "спортивное",
            "обычное",
            "мускулистое",
            "худощавое",
            missing,
        ],
        "Цвет волос": [
            "светлые",
            "темные",
            "седые",
            "яркие",
            "мелированные",
            "бритый наголо",
            "рыжие",
            missing,
        ],
        "Цвет глаз": [
            "зеленые",
            "голубые",
            "карие",
            "серые",
            "другие",
            missing,
        ],
    }
    multicategory_features = {
        "Знакомлюсь для": [
            "флирта",
            "любви и романтики",
            "cерьезных отношений",
            "переписки",
            "другого",
        ],
        "Знание языков": [
            "русский",
            "английский",
            "немецкий",
            "французский",
            "испанский",
            "итальянский",
        ],
    }

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

        self._scaler = None
        self._onehot_encoder = None
        self._multilabel_encoders = {}
        self._numeric_imputer = None
        self._knn_imputer = None
        self._feature_names = None

    def fit(self, X: pd.DataFrame, y=None):
        X_fit = X.copy()

        for col in self.categorical_features:
            X_fit[col] = X_fit[col].fillna(self.missing)

        self._onehot_encoder = OneHotEncoder(
            categories=list(self.categorical_features.values()),
            drop="first",
            sparse_output=False,
        )
        dummy_categorical_data = pd.DataFrame(
            {
                col: [self.categorical_features[col][0]]
                for col in self.categorical_features
            }
        )
        self._onehot_encoder.fit(dummy_categorical_data)

        for col in self.multicategory_features:
            mlb = MultiLabelBinarizer(classes=self.multicategory_features[col])
            mlb.fit([])
            self._multilabel_encoders[col] = mlb

        all_numeric_data = self._transform_to_numeric(X_fit)

        self._scaler = StandardScaler()
        scaled_data = self._scaler.fit_transform(all_numeric_data)

        self._knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)
        self._knn_imputer.fit(scaled_data)

        self._feature_names = all_numeric_data.columns.to_list()

        return self

    def transform(self, X: pd.DataFrame, y=None):
        X_transform = X.copy()

        for col in self.categorical_features:
            X_transform[col] = X_transform[col].fillna(self.missing)

        all_numeric_data = self._transform_to_numeric(X_transform)
        scaled_data = self._scaler.transform(all_numeric_data)
        imputed_scaled = self._knn_imputer.transform(scaled_data)

        all_numeric_data[self.numeric_features] = imputed_scaled[
            :, : len(self.numeric_features)
        ]

        return pd.DataFrame(all_numeric_data, columns=self._feature_names)

    def _transform_to_numeric(self, X: pd.DataFrame):
        numeric_parts = []

        numeric_df = X[self.numeric_features].copy()
        numeric_parts.append(numeric_df.reset_index(drop=True))

        encoded = self._onehot_encoder.transform(X[self.categorical_features.keys()])
        encoded_df = pd.DataFrame(
            encoded, columns=self._onehot_encoder.get_feature_names_out()
        )
        numeric_parts.append(encoded_df)

        for col in self.multicategory_features:
            mlb = self._multilabel_encoders[col]
            encoded = mlb.transform(
                X[col].apply(lambda s: s.split(", ") if pd.notna(s) else [])
            )
            encoded_df = pd.DataFrame(
                encoded, columns=[f"{col}_{tag}" for tag in mlb.classes_]
            )
            numeric_parts.append(encoded_df)

        return pd.concat(numeric_parts, axis=1)

    def get_feature_names_out(self, input_features=None):
        return self._feature_names
