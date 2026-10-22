from django.shortcuts import render
import pandas as pd
import time
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
 
from aplicativosML.models import Dados_Apontamentos
 
def executarRegressaoLogistica(request):
    executado = 'divisao' in request.GET or 'algoritmo' in request.GET
    algoritmo_selecionado = request.GET.get('algoritmo', 'ambos')
    if algoritmo_selecionado not in ['ambos', 'nb', 'rl']:
        algoritmo_selecionado = 'ambos'
    
    # Slider alinhado ao template (60 a 90 de treino)
    try:
        pct_treino = int(request.GET.get('divisao', '80'))
        if pct_treino < 60 or pct_treino > 90:
            pct_treino = 80
    except (ValueError, TypeError):
        pct_treino = 80
 
    pct_teste = 100 - pct_treino
    test_size_val = float(pct_teste) / 100.0
 
    acc_rl, prec_rl, rec_rl, f1_rl = 0.0, 0.0, 0.0, 0.0
    acc_nb, prec_nb, rec_nb, f1_nb = 0.0, 0.0, 0.0, 0.0
    time_nb, time_rl = 0.0, 0.0
 
    if executado:
        campos = ['cidade', 'codigo_cartorio', 'valor_original', 'valor_enviado', 'status']
        dados = Dados_Apontamentos.objects.values(*campos)
        df_base = pd.DataFrame.from_records(dados)
 
        if not df_base.empty and 'status' in df_base.columns:
            df_base.columns = df_base.columns.str.strip()
 
            # Normalização do Target
            df_base['status'] = df_base['status'].replace('RECEBIDO PELO CARTÓRIO', 'PROTESTADO')
            df_base['status'] = df_base['status'].replace('PAGO PELO DEVEDOR', 'REPASSADO AO CREDOR')
            df_base['status'] = df_base['status'].replace('RETIRADO', 'CANCELADO')
            y = df_base['status'].apply(lambda x: 1 if str(x).strip().upper() == 'PROTESTADO' else 0)
 
            # Preparação das Features
            df_features = df_base.drop('status', axis=1).copy()
 
            # Conversão segura de valores monetários para float
            for col_val in ['valor_original', 'valor_enviado']:
                if col_val in df_features.columns:
                    df_features[col_val] = pd.to_numeric(df_features[col_val], errors='coerce').fillna(0.0)
 
            # Encodings de variáveis categóricas
            for col in df_features.select_dtypes(include=['object']).columns:
                df_features[col] = LabelEncoder().fit_transform(df_features[col].astype(str))
 
            # Split único e justo para ambos os modelos (random_state=42)
            strat = y if y.value_counts().min() >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(
                df_features, y, test_size=test_size_val, random_state=42, stratify=strat
            )
 
            # 1. Regressão Logística
            if algoritmo_selecionado in ['ambos', 'rl']:
                t_inicio = time.time()
                smote = SMOTE(random_state=42)
                X_res, y_res = smote.fit_resample(X_train, y_train)
 
                scaler = StandardScaler()
                X_res_scaled = scaler.fit_transform(X_res)
                X_test_scaled = scaler.transform(X_test)
 
                model_rl = LogisticRegression(max_iter=1000, class_weight='balanced')
                model_rl.fit(X_res_scaled, y_res)
                y_pred_rl = model_rl.predict(X_test_scaled)
                time_rl = round((time.time() - t_inicio) * 1000, 1)
 
                acc_rl = float(round(accuracy_score(y_test, y_pred_rl) * 100, 2))
                prec_rl = float(round(precision_score(y_test, y_pred_rl, zero_division=0) * 100, 2))
                rec_rl = float(round(recall_score(y_test, y_pred_rl, zero_division=0) * 100, 2))
                f1_rl = float(round(f1_score(y_test, y_pred_rl, zero_division=0) * 100, 2))
 
            # 2. Naive Bayes (avaliado exatamente sobre o mesmo X_test e y_test)
            if algoritmo_selecionado in ['ambos', 'nb']:
                t_inicio = time.time()
                # MultinomialNB exige valores não-negativos
                X_train_nb = X_train.clip(lower=0)
                X_test_nb = X_test.clip(lower=0)
 
                modelo_nb = MultinomialNB()
                modelo_nb.fit(X_train_nb, y_train)
                y_pred_nb = modelo_nb.predict(X_test_nb)
                time_nb = round((time.time() - t_inicio) * 1000, 1)
 
                acc_nb = float(round(accuracy_score(y_test, y_pred_nb) * 100, 2))
                prec_nb = float(round(precision_score(y_test, y_pred_nb, zero_division=0) * 100, 2))
                rec_nb = float(round(recall_score(y_test, y_pred_nb, zero_division=0) * 100, 2))
                f1_nb = float(round(f1_score(y_test, y_pred_nb, zero_division=0) * 100, 2))
 
    contexto = {
        'executado': executado,
        'algoritmo_selecionado': algoritmo_selecionado,
        'pct_treino': pct_treino,
        'pct_teste': pct_teste,
        'time_nb': time_nb,
        'time_rl': time_rl,
        'acc_nb': acc_nb,
        'prec_nb': prec_nb,
        'rec_nb': rec_nb,
        'f1_nb': f1_nb,
        'acc_rl': acc_rl,
        'prec_rl': prec_rl,
        'rec_rl': rec_rl,
        'f1_rl': f1_rl,
    }
 
    return render(request, "resultado.html", contexto)
 