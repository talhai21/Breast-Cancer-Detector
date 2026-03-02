import zipfile, pickle
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    FEATURE_NAMES = [
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
    "mean_compactness",
    "mean_concavity",
    "mean_concave_points",
    "mean_symmetry",
    "mean_fractal_dimension",
    "radius_error",
    "texture_error",
    "perimeter_error",
    "area_error",
    "smoothness_error",
    "compactness_error",
    "concavity_error",
    "concave_points_error",
    "symmetry_error",
    "fractal_dimension_error",
    "worst_radius",
    "worst_texture",
    "worst_perimeter",
    "worst_area",
    "worst_smoothness",
    "worst_compactness",
    "worst_concavity",
    "worst_concave_points",
    "worst_symmetry",
    "worst_fractal_dimension"
]

    def load_wdbc_from_zip(zip_path: str) -> pd.DataFrame:
        with zipfile.ZipFile(zip_path) as z:
            raw = z.read('wdbc.data').decode('utf-8').strip().splitlines()
        rows=[r.split(',') for r in raw if r.strip()]
        data=np.array(rows, dtype=object)
        cols=['id','diagnosis']+[f'feat_{i}' for i in range(1,31)]
        df=pd.DataFrame(data, columns=cols)
        for c in cols[2:]:
            df[c]=df[c].astype(float)
        df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})
        df.rename(columns={c:n for c,n in zip(cols[2:], FEATURE_NAMES)}, inplace=True)
        return df

    def train(df: pd.DataFrame, out_path: str='model.pkl') -> None:
        X=df[FEATURE_NAMES].values
        y=df['diagnosis'].values
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        scaler=StandardScaler()
        X_train_s=scaler.fit_transform(X_train)
        X_test_s=scaler.transform(X_test)

        model=LogisticRegression(max_iter=5000, solver='lbfgs')
        model.fit(X_train_s,y_train)
        pred=model.predict(X_test_s)
        proba=model.predict_proba(X_test_s)[:,1]
        acc=accuracy_score(y_test,pred)
        auc=roc_auc_score(y_test,proba)
        print(f"Test accuracy: {acc:.4f}")
        print(f"Test ROC AUC:  {auc:.4f}")

        bundle={'scaler': scaler, 'model': model, 'feature_names': FEATURE_NAMES}
        with open(out_path,'wb') as f:
            pickle.dump(bundle,f)
        print(f"Saved {out_path}")

    if __name__ == '__main__':
        import argparse
        p=argparse.ArgumentParser()
        p.add_argument('--zip', required=True, help='Path to breast-cancer-wisconsin-original.zip')
        p.add_argument('--out', default='model.pkl', help='Output model path')
        args=p.parse_args()
        df=load_wdbc_from_zip(args.zip)
        train(df, args.out)